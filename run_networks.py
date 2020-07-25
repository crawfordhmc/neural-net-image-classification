import tensorflow as tf

# this script loads the most successful pre-trained networks

simple_clean_model = tf.keras.models.load_model('task1_bnorm.h5')
simple_noisy_model = tf.keras.models.load_model('task1_noisy_drop_bnorm.h5')
fine_clean_model = tf.keras.models.load_model('task2_clean_drop_wd_bnorm.h5')
fine_noisy_model = tf.keras.models.load_model('task2_noisy_drop_wd_bnorm.h5')