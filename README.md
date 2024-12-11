# dl-creating-deep-learning-net-12-24
---
Abaixo há a explicação detalhada do código de criação de uma rede de deep learning para classificação de dígitos escritos à mão usando o conjunto de dados MNIST:

1. **Configuração Inicial e Importações**:
   - O script começa instalando o TensorFlow
   - Importa bibliotecas essenciais para machine learning:
     - Matplotlib para visualização
     - TensorFlow e Keras para construção da rede neural
     - Módulos específicos para camadas de rede, otimização e callbacks

2. **Preparação dos Dados**:
   ```python
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   X_train = x_train / 255
   x_test = x_test / 255
   ```
   - Carrega o dataset MNIST de dígitos manuscritos
   - Normaliza os pixels dividindo por 255, transformando valores de 0-255 para 0-1
   - Isso ajuda a estabilizar o treinamento, tornando o modelo menos sensível a variações menores

3. **Arquitetura da Rede Neural**:
   ```python
   model = Sequential()
   model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(28,28,1)))
   model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))
   model.add(Flatten())
   model.add(Dense(128, activation='relu'))
   model.add(Dropout(0.5))
   model.add(Dense(10, activation='softmax'))
   ```
   - Usa uma arquitetura de Rede Neural Convolucional (CNN)
   - Camadas:
     - Primeira camada convolucional com 32 filtros
     - Segunda camada convolucional com 64 filtros
     - Camada de max pooling para reduzir dimensionalidade
     - Dropout para prevenir overfitting
     - Flatten para achatar a entrada
     - Camada densa intermediária com 128 neurônios
     - Camada final com 10 neurônios (para 10 dígitos) com softmax para classificação

4. **Compilação do Modelo**:
   ```python
   optimizer = Adam()
   model.compile(optimizer=optimizer, 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
   ```
   - Usa otimizador Adam
   - Função de perda categorical crossentropy (ideal para classificação multi-classe)
   - Métrica de acurácia para avaliação

5. **Configurações de Treinamento**:
   ```python
   learning_rate_reduction = ReduceLROnPlateau(
       monitor='val_acc',
       patience=3,
       verbose=1,
       factor=0.5,
       min_lr=0.00001
   )
   ```
   - Callback para reduzir learning rate se não houver melhora na acurácia de validação
   - Ajuda a encontrar o mínimo global de forma adaptativa

6. **Preparação dos Rótulos**:
   ```python
   y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
   y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
   ```
   - Converte rótulos para one-hot encoding
   - Necessário para classificação multi-classe

7. **Treinamento do Modelo**:
   ```python
   history = model.fit(x_train, y_train,
                       batch_size=32,
                       epochs=10,
                       validation_split=0.2,
                       verbose=1,
                       callbacks=[learning_rate_reduction])
   ```
   - Treina o modelo por 10 épocas
   - Usa 20% dos dados para validação
   - Aplica o callback de redução de learning rate

8. **Visualização**:
   ```python
   plt.plot(range_epochs, val_acc, label='Acurácia no conjunto de validação')
   plt.plot(range_epochs, acc, label='Acurácia no conjunto de treinamento')
   ```
   - Plota a acurácia de treinamento e validação ao longo das épocas
   - Ajuda a visualizar o progresso do aprendizado e potencial overfitting

O código demonstra uma implementação completa de uma rede neural convolucional para reconhecimento de dígitos manuscritos, incluindo preparação de dados, construção do modelo, treinamento e visualização de resultados.
