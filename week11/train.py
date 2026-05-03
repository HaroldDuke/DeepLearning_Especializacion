import os
import tensorflow as tf


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=50, batch_size=32, save_dir='models'):
    # Creamos la carpeta de modelos si no existe
    os.makedirs(save_dir, exist_ok=True)

    # Ruta donde se guardará el mejor checkpoint de este modelo
    model_path = os.path.join(save_dir, f'{model.name}_best.keras')

    callbacks = [
        # Detiene el entrenamiento si val_loss no mejora en 5 épocas consecutivas
        # y restaura los pesos de la mejor época automáticamente
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),

        # Guarda el modelo completo solo cuando mejora val_loss
        # así siempre tenemos guardado el mejor estado visto hasta ese momento
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_loss', save_best_only=True, verbose=0
        ),

        # Si val_loss se estanca 3 épocas seguidas, reduce la tasa de aprendizaje a la mitad
        # Esto permite salir de mesetas sin bajar el LR desde el inicio
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0
        ),
    ]

    # Entrenamos el modelo con los datos de entrenamiento y validamos en cada época
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,  # número de muestras procesadas antes de actualizar pesos
        callbacks=callbacks,
        verbose=1
    )

    return history
