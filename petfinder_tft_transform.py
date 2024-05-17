import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = ['AdoptionSpeed']
CAT_FEATURES = ['Type', 'Gender', 'Color1', 'Color2', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health']
NUMERICAL_FEAT = ['Age', 'Fee', 'PhotoAmt']


def transformed_name(name):
    return name + "_xf"


def binarize_adoption_speed_tfop(input_value):
    """
    Binarize the adoption speed
    """
    return tf.reshape(tf.where(input_value == 4, 1.0, 0.0), [-1])


def cat_to_one_hot_tfop(input_value, vocab_filename='x_vocab'):
    integerized = tft.compute_and_apply_vocabulary(
        input_value,
        num_oov_buckets=1,
        vocab_filename=vocab_filename)
    one_hot_encoded = tf.one_hot(
        integerized,
        depth=tf.cast(tft.experimental.get_vocabulary_size_by_name(vocab_filename) + 1,
                      tf.int32),
        on_value=1.0,
        off_value=0.0)
    return one_hot_encoded


def preprocessing_fn(inputs):
    outputs = {}
    for feat in CAT_FEATURES:
        outputs[feat] = cat_to_one_hot_tfop(inputs[feat], vocab_filename="petfinder" + feat)
    for feat in NUMERICAL_FEAT:
        outputs[feat] = tft.scale_to_z_score(inputs[feat])
    for feat in LABEL_KEY:
        outputs[feat] = binarize_adoption_speed_tfop(inputs[feat])

    return outputs
