import os
import time
import logging
import argparse
import paddle
import numpy as np
from paddle import fluid
from tqdm import tqdm

from module import Dataset, Evaluator, SModel, OPModel, AdamW, LinearDecay, Tokenizer, at_loss

parser = argparse.ArgumentParser(description="Main Program.")
parser.add_argument('--name', type=str, required=True, help="Experiment on which dataset ?")
parser.add_argument('--partial', type=str, default="PM", help="Evaluate on partial match ?")
parser.add_argument('--glove_path', type=str, default='glove/glove.6B.300d.txt', help="GloVe word vectors path")
parser.add_argument('--glove_dim', type=int, default=300, help="Glove word embedding dimension.")
parser.add_argument('--train_path', type=str, required=True, help="Training corpus path.")
parser.add_argument('--valid_path', type=str, required=True, help="Validation corpus path.")
parser.add_argument('--test_path', type=str, required=True, help="Testing corpus path.")
parser.add_argument('--schemas_path', type=str, required=True, help="Auxiliary information.")

parser.add_argument('--num_relations', type=int, required=True, help="Total number of relation types.")
parser.add_argument('--num_subs', type=int, required=True, help="Total number of subject types.")
parser.add_argument('--num_objs', type=int, required=True, help="Total number of object types.")
parser.add_argument('--max_len', type=int, default=100, help="Max sequence length.")
parser.add_argument('--hidden_size', type=int, default=128, help="Hidden size.")
parser.add_argument("--type_size", type=int, default=64, help="Subject type embedding size.")
parser.add_argument("--epoch", type=int, default=60, help="Total epoch.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--device", type=str, default='cpu', help="CUDA or CPU ?")

args = parser.parse_args()

MODE = args.mode
NAME = args.name
PLACE = fluid.CPUPlace() if args.device == 'cpu' else fluid.CUDAPlace(int(args.device))
LOG = NAME + '.log'


def main():
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(filename=LOG, encoding='utf-8'))
    with fluid.dygraph.guard(PLACE):
        tokenizer = Tokenizer(args.glove_path, args.glove_dim)
        dataset = Dataset(args.train_path, args.schemas_path, tokenizer, args.max_len, args.batch_size, NAME)
        s_model = SModel(args.glove_dim, args.hidden_size, args.max_len, args.num_subs)
        op_model = OPModel(args.hidden_size, args.type_size, args.max_len, args.num_relations, args.num_subs)
        parameter_list = s_model.parameters() + op_model.parameters()
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        for p in parameter_list:
            mul_value = np.prod(p.shape)
            total_params += mul_value
            if p.stop_gradient:
                non_trainable_params += mul_value
            else:
                trainable_params += mul_value
        print(f'Total params: {total_params}')
        print(f'Trainable params: {trainable_params}')
        print(f'Non-trainable params: {non_trainable_params}')
        if NAME == "NYT10":
            opt = AdamW(learning_rate=LinearDecay(1e-3, 3000, 150000), parameter_list=parameter_list, weight_decay=0.01)
        elif NAME == "NYT11":
            opt = AdamW(learning_rate=LinearDecay(1e-3, 2000, 120000), parameter_list=parameter_list, weight_decay=0.01)
        elif NAME == "NYT24":
            opt = AdamW(learning_rate=LinearDecay(1e-3, 2000, 120000), parameter_list=parameter_list, weight_decay=0.01)
        else:
            print("WRONG NAME. TRY AGAIN.")
            return
        evaluator = Evaluator(dataset, NAME)
        # Train
        for epoch in range(args.epoch):
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"\nEpoch {epoch + 1} starts at {time_stamp}. The lr is {opt.current_step_lr()}.")
            s_model.train()
            op_model.train()
            data_loader = fluid.io.DataLoader.from_generator(capacity=64, return_list=True)
            data_loader.set_batch_generator(dataset.generate_batch(), places=PLACE)
            step = 0
            epoch_loss, epoch_sub_loss, epoch_obj_loss = 0.0, 0.0, 0.0
            batch_loss, batch_sub_loss, batch_obj_loss = 0.0, 0.0, 0.0
            start_time = time.time()
            for batch_id, batch_data in tqdm(enumerate(data_loader())):
                token_ids, sent_vec, sub_heads, sub_tails, sub_id, sub_head, sub_tail, obj_heads, obj_tails = batch_data
                token_ids = fluid.dygraph.to_variable(token_ids).astype("int64")
                sent_vec = fluid.dygraph.to_variable(sent_vec).astype("float32")
                sub_heads = fluid.dygraph.to_variable(sub_heads).astype("float32")
                sub_tails = fluid.dygraph.to_variable(sub_tails).astype("float32")
                sub_id = fluid.dygraph.to_variable(sub_id).astype("int64")
                sub_head = fluid.dygraph.to_variable(sub_head).astype("int64")
                sub_tail = fluid.dygraph.to_variable(sub_tail).astype("int64")
                obj_heads = fluid.dygraph.to_variable(obj_heads).astype("float32")
                obj_tails = fluid.dygraph.to_variable(obj_tails).astype("float32")
                s_h, s_t, features, mask = s_model(token_ids, sent_vec)
                o_h, o_t = op_model(features, sub_id, sub_head, sub_tail, mask)
                sub_loss = at_loss(s_h, sub_heads, mask) + at_loss(s_t, sub_tails, mask)
                obj_loss = at_loss(o_h, obj_heads, mask) + at_loss(o_t, obj_tails, mask)
                loss = sub_loss + obj_loss
                epoch_loss += loss.numpy()
                batch_loss += loss.numpy()
                epoch_sub_loss += sub_loss.numpy()
                batch_sub_loss += sub_loss.numpy()
                epoch_obj_loss += obj_loss.numpy()
                batch_obj_loss += obj_loss.numpy()
                step += 1
                if batch_id > 0 and batch_id % 500 == 0:
                    average_loss = round(float(batch_loss / 500), 4)
                    average_sub_loss = round(float(batch_sub_loss / 500), 4)
                    average_obj_loss = round(float(batch_obj_loss / 500), 4)
                    logger.info(f'\nEpoch:{epoch + 1}: complete {batch_id} batches '
                                f' loss:{average_loss} '
                                f' subs: {average_sub_loss} '
                                f' objs: {average_obj_loss} ')
                    batch_loss, batch_sub_loss, batch_obj_loss = 0.0, 0.0, 0.0
                loss.backward()
                opt.minimize(loss)
                s_model.clear_gradients()
                op_model.clear_gradients()
            end_time = time.time()
            used_time = end_time - start_time
            average_epoch_loss = round(float(epoch_loss / step), 5)
            average_epoch_sub_loss = round(float(epoch_sub_loss / step), 4)
            average_epoch_obj_loss = round(float(epoch_obj_loss / step), 4)
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"\nEpoch {epoch + 1} ends at {time_stamp},"
                        f" used time: {int(used_time)}s ,"
                        f" loss:{average_epoch_loss},"
                        f" subs:{average_epoch_sub_loss}, "
                        f" objs:{average_epoch_obj_loss} ")
            if average_epoch_loss < 1.0:
                time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"\nEvaluation starts at Epoch {epoch + 1}, {time_stamp}")
                s_model.eval()
                op_model.eval()
                f1, p, r, best_f1, used_time = evaluator.evaluate(args.valid_path, s_model, op_model, args.partial)
                logger.info(f'\nEpoch:{epoch + 1}\t F1: {f1} \t P: {p} \t R: {r} \t best_F1: {best_f1}, '
                            f' used time: {used_time}s')
                s_model.clear_gradients()
                op_model.clear_gradients()
        # Test
        logger.info(f"\nTest starts at {time_stamp}")
        load_path = os.path.join(os.getcwd(), f"{args.test_path.split('/')[2]}_best_model")
        sub_load_path = os.path.join(load_path, "SModel")
        obj_load_path = os.path.join(load_path, "OPModel")
        sub_state_dict = fluid.load_dygraph(sub_load_path)
        obj_state_dict = fluid.load_dygraph(obj_load_path)
        s_model.set_state_dict(sub_state_dict)
        op_model.set_state_dict(obj_state_dict)
        s_model.eval()
        op_model.eval()
        f1, p, r, _, used_time = evaluator.evaluate(args.test_path, s_model, op_model, args.partial)
        logger.info(f'\nTest Result\t F1: {f1} \t P: {p} \t R: {r}'
                    f' used time: {used_time}s')
        s_model.clear_gradients()
        op_model.clear_gradients()


if __name__ == "__main__":
    main()
