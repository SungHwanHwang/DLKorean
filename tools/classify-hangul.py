#!/usr/bin/env python
import argparse
import io
import os
import tensorflow as tf

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# label파일/model파일/test_image파일로 인식률 테스트 진행
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, '../label/2350-common-hangul.txt')
DEFAULT_GRAPH_FILE = os.path.join(SCRIPT_PATH, '../saved-model/optimized_hangul_tensorflow.pb')
DEFAULT_TEST_FILE = os.path.join(SCRIPT_PATH, '../test/')


def read_image(file):
    """Read an image file and convert it into a 1-D floating point array."""
    file_content = tf.read_file(file)
    image = tf.image.decode_jpeg(file_content, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [64*64])
    return image


def classify(args):
    """
    가장 높은 다섯가지 예측을 보여줌
    """
    labels = io.open(args.label_file,'r', encoding='utf-8').read().splitlines()

    #if not os.path.isfile(args.test_file):
    #    print('Error: Image %s not found.' % args.test_file)
    #    sys.exit(1)

    # Load graph and parse file.
    with tf.gfile.GFile(args.graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='hangul-model',
            producer_op_list=None
        )

    x = graph.get_tensor_by_name('hangul-model/input:0')
    y = graph.get_tensor_by_name('hangul-model/output:0')
    keep_prob = graph.get_tensor_by_name('hangul-model/keep_prob:0')

    # test할 이미지들의 경로
    list = os.listdir(args.test_file)

    for i in range(len(list)):
        image = read_image(args.test_file + list[i])
        sess = tf.InteractiveSession()
        image_array = sess.run(image)
        sess.close()
        with tf.Session(graph=graph) as graph_sess:
            predictions = graph_sess.run(y, feed_dict={x: image_array, keep_prob: 1.0})
            prediction = predictions[0]

        # 예측률 중 가장 높은 5가지 출력
        sorted_indices = prediction.argsort()[::-1][:5]
        for index in sorted_indices:
            label = labels[index]
            confidence = prediction[index]
            print('%s (confidence = %.5f)' % (label, confidence))
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, dest='test_file', default=DEFAULT_TEST_FILE,
                        help='Image to pass to model for classification.')
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--graph-file', type=str, dest='graph_file',
                        default=DEFAULT_GRAPH_FILE,
                        help='The saved model graph file to use for '
                             'classification.')
    classify(parser.parse_args())
