from tensorflow.keras.callbacks import EarlyStopping

def train_autoencoder(
    autoencoder,
    X_training_set
):
    autoencoder.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=3, min_delta= 0.0001, restore_best_weights=True)

    history = autoencoder.fit(X_training_set, X_training_set, epochs=100, batch_size=256, shuffle=True, validation_split=0.2, callbacks=[early_stop])

    return history