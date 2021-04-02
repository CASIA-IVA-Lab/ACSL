import logging
import os.path
import pickle
import numpy as np
from lvis import LVIS, LVISResults, LVISEval
import argparse

parser = argparse.ArgumentParser(description='MMDet test detector')
parser.add_argument('--boxjson', help='test config file path')
parser.add_argument('--segjson', type=str, default='None')
parser.add_argument('--result_path', type=str)
args = parser.parse_args()

class LVISEvalPerCls(LVISEval):

    def __init__(self, lvis_gt, lvis_dt, iou_type="segm", result_path=None):
        super(LVISEvalPerCls, self).__init__(lvis_gt, lvis_dt, iou_type)
        self.result_path = result_path

    def _summarize(
        self, summary_type, iou_thr=None, area_rng="all", cat_id=None
    ):
        assert cat_id is not None
        aidx = [
            idx
            for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]

        if summary_type == 'ap':
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if cat_id is not None:
                s = s[:, :, cat_id, aidx]
            else:
                s = s[:, :, :, aidx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, :, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        precision = []
        for i in range(1230):
            precision.append(self._summarize('ap', cat_id=i))

        lines = []
        for p in precision:
            line = '{}\n'.format(p)
            lines.append(line)

        with open(self.result_path, 'w') as f:
            f.writelines(lines)

        print('The result file has been saved to {}'.format(self.result_path))

def get_split_bin():
    split_file_name = './data/lvis/valsplit.pkl'

    if os.path.exists(split_file_name):
        with open(split_file_name, 'rb') as fin:
            splits = pickle.load(fin)
        print('Load split file from: {}'.format(split_file_name))
        return splits

def accumulate_acc(num_ins, num_get, splitbin):

    print('Saving pro cls result to: {}'.format('tempcls.pkl'))

    print('\n')
    print('========================================================')
    title_format = "| {} | {} | {} | {} | {} | {} |"
    print(title_format.format('Type', 'IoU', 'Area', 'MaxDets', 'CatIds',
                              'Result'))
    print(title_format.format(':---:', ':---:', ':---:', ':---:', ':---:',
                              ':---:'))
    template = "| {:^6} | {:<9} | {:>6s} | {:>3d} | {:>12s} | {:2.2f}% |"
    for k, v in splitbin.items():
        ins_count = num_ins[v].sum().astype(np.float64)
        get_count = num_get[v].sum().astype(np.float64)
        acc = get_count / ins_count
        print(template.format('(ACC)', '0.50:0.95', 'all', 300, k, acc * 100))


# with open('tempcls.pkl', 'rb') as fin:
#     savelist = pickle.load(fin)

# num_get = savelist[0]
# num_ins = savelist[1]
# splitbin = get_split_bin()
# accumulate_acc(num_ins, num_get, splitbin)

# result and val files for 100 randomly sampled images.
ANNOTATION_PATH = "data/lvis/lvis_v0.5_val.json"

RESULT_PATH_BBOX = args.boxjson
print('Eval Bbox:')
ANN_TYPE = 'bbox'
lvis_eval = LVISEvalPerCls(ANNOTATION_PATH, RESULT_PATH_BBOX, ANN_TYPE, args.result_path)
lvis_eval.run()
lvis_eval.print_results()

if not args.segjson == 'None':
    RESULT_PATH_SEGM = args.segjson
    print('Eval Segm:')
    ANN_TYPE = 'segm'
    lvis_eval = LVISEval(ANNOTATION_PATH, RESULT_PATH_SEGM, ANN_TYPE)
    lvis_eval.run()
    lvis_eval.print_results()
