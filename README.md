# Perceptrons-Multi-Camada
# MLP para Resolver o Problema do XOR

## Descrição
Este projeto implementa um MLP (Perceptron Multicamadas) para resolver o problema da porta lógica XOR. O MLP é treinado utilizando o algoritmo de backpropagation e a função de ativação sigmoide.

## Estrutura do Código
O código é dividido em várias partes:
1. **Definição da Classe MLP**: Contém a implementação do MLP com uma camada escondida.
2. **Funções de Ativação e Custo**: Inclui a função sigmoide e sua derivada, bem como a função de erro quadrático médio (MSE).
3. **Passo Feedforward e Backpropagation**: Implementa o processo de feedforward para calcular a saída da rede e o processo de backpropagation para ajustar os pesos com base nos erros.
4. **Treinamento e Teste**: Treina a rede neural com dados da porta XOR e testa a rede com os mesmos dados.

### Implementação

#### Definição da Classe MLP
```python
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Pesos da camada de entrada para a camada escondida
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        
        # Pesos da camada escondida para a camada de saída
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))
```
#### Funções de Ativação e Custo
```python
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def _mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
```
#### Passo Feedforward e Backpropagation
```python
    def forward_pass(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self._sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self._sigmoid(self.final_input)
        return self.final_output
    
    def backward_pass(self, X, y, output):
        error = y - output
        d_output = error * self._sigmoid_derivative(output)
        
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * self._sigmoid_derivative(self.hidden_output)
        
        # Atualização dos pesos e bias
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate
```
#### Treinamento e Teste
```python
    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward_pass(X)
            self.backward_pass(X, y, output)
            if (epoch + 1) % 1000 == 0:
                mse = self._mse(y, output)
                print(f'Epoch {epoch + 1}, MSE: {mse}')

# Dados de entrada e saída para a porta XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Criação e treinamento do MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
mlp.train(X, y)

# Testando a rede
for x in X:
    print(f'Input: {x}, Predicted Output: {mlp.forward_pass(x)}')
```
### Instalação
Para rodar este projeto no Google Colab, você não precisa instalar nada localmente. Basta acessar o [Google Colab](https://colab.google/), criar um novo notebook e colar o código fornecido.

### Demonstração
[Link para o vídeo](https://drive.google.com/file/d/1eCFuJTuAVi3url3CPVegMPEMmHmglfd_/view?usp=drivesdk)

### Link para o Google Colab
[Link para o Google Colab](https://colab.research.google.com/drive/1rEneYO4GpAg8zDDHs38nLR2PGbKrK_kV?usp=sharing)
