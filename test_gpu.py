import tensorflow as tf

# Listar todos os dispositivos disponíveis
devices = tf.config.list_physical_devices()
print("Todos os dispositivos disponíveis:")
for device in devices:
    print(device)

# Verificar se há GPUs disponíveis
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("\nGPUs disponíveis:")
    for gpu in gpu_devices:
        print(gpu)
else:
    print("\nNenhuma GPU disponível.")

# Verificar se TensorFlow está usando a GPU
print("\nTensorFlow está usando a GPU?")
print("Sim" if tf.config.experimental.list_physical_devices('GPU') else "Não")
