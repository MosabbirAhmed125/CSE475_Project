from tensorflow import keras
from tensorflow.keras import layers


def build_efficientnet(
    num_classes=6,
    input_shape=(150, 150, 3)
):

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = keras.Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
