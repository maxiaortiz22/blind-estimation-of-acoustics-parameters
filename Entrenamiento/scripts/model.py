import tensorflow as tf
import tensorflow.keras.layers as tfkl


def modelo():
    tf.keras.backend.clear_session()

    audio_in = tfkl.Input((200,1), name = 'Audio de entrada')

    capa_1 = tfkl.Conv1D(filters=32, kernel_size=(10), activation='relu', name='capa1_conv')(audio_in)
    capa_1 = tfkl.MaxPool1D(pool_size=2, name='capa1_pool')(capa_1)
    capa_1 = tfkl.BatchNormalization(name = 'Batch_capa1')(capa_1)

    capa_2 = tfkl.Conv1D(filters=16, kernel_size=(5), activation='relu', name='capa2_conv')(capa_1) 
    capa_2 = tfkl.MaxPool1D(pool_size=2, name='capa2_pool')(capa_2)
    capa_2 = tfkl.BatchNormalization(name = 'Batch_capa2')(capa_2)
    capa_2 = tfkl.Dropout(0.4, name='capa2_drop')(capa_2)

    capa_3 = tfkl.Conv1D(filters=8, kernel_size=(5), activation='relu', name='capa3_conv')(capa_2) 
    capa_3 = tfkl.MaxPool1D(pool_size=2, name='capa3_pool')(capa_3)
    capa_3 = tfkl.BatchNormalization(name = 'Batch_capa3')(capa_3)
    
    capa_4 = tfkl.Conv1D(filters=4, kernel_size=(5), activation='relu', name='capa4_conv')(capa_3)
    capa_4 = tfkl.Flatten()(capa_4)
    tr_pred = tfkl.Dense(1, name='Salida_prediccion')(capa_4) # solo calculo tr

    modelo = tf.keras.Model(inputs=[audio_in], outputs=[tr_pred])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    return modelo