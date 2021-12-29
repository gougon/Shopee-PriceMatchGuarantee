import pandas as pd
import category_encoders as ce

import config.constant as const


class Preprocessing():
    def __init__(self, encoding_cols):
        self.__encoder = ce.BinaryEncoder(cols=encoding_cols, return_df=True)

    def proc(self, raw_train):
        data_encoded = self.__encoder.fit_transform(raw_train)
        return data_encoded


if __name__ == '__main__':
    raw_train = pd.read_csv(const.DATA_DIR + const.RAW_DATA_TRAIN)
    preprocessing = Preprocessing([const.LABEL_COL_NAME])
    proc_train = preprocessing.proc(raw_train)
    proc_train.to_csv(const.DATA_DIR + const.PROC_DATA_TRAIN, index=False)
