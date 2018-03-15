import numpy as np
import tensorflow as tf


N = 4000
STEPS = 5000


if __name__ == '__main__':
    image = []
    label_tmp = []

    for i in range(1, N+1):
        feature_4x = np.load("feature/feature_4x_"+str(i)+".npy")
        label = np.load("label/label_"+str(i)+".npy")
        image.append(feature_4x[0, :])
        label_tmp.append(label[0, :])
        print(i)

    x = np.array(image)
    label = np.array(label_tmp)

    train_length = int(0.95*len(x))

    train_x = x[:train_length]
    train_label = label[:train_length]

    test_x = x[train_length:]
    test_label = label[train_length:]

    hidden_layer_size = 50

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[75])]

    # Build a DNNRegressor.
    model = tf.estimator.DNNRegressor(label_dimension=3,
                                      feature_columns=feature_columns,
                                      hidden_units=[hidden_layer_size],
                                      model_dir="init_model")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_label,
        batch_size=128,
        num_epochs=None,
        shuffle=True)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_x},
        y=test_label,
        batch_size=128,
        num_epochs=1,
        shuffle=False)

    # Train the model.
    model.train(input_fn=train_input_fn, steps=STEPS)

    eval_result = model.evaluate(input_fn=test_input_fn)
    average_loss = eval_result["average_loss"]

    print(average_loss)
