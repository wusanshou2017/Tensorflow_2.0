import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Position_Embedding():
    def __init__(self):
        pass

    def get_angles(self, pos, i, d_model=512):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model=512):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # 将 sin 应用于数组中的偶数索引（indices）；2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 将 cos 应用于数组中的奇数索引；2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


if __name__ == '__main__':

    pos_emb = Position_Embedding()
    res = pos_emb.positional_encoding(50, 512)
    print(res.shape)

    plt.pcolormesh(res[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
