import sklearn as sk
import sklearn.model_selection
import math





def load_and_preprocess_data():
    dataset = sk.datasets.load_digits()
    X, y = dataset.data, dataset.target
    return train_test_split(X, y, test_size=0.2, random_state=42)
def main() -> None:
    return
def relu(x) -> int:
    if(x < 0):
        return 0
    else:
        return x


def gradient_cliping(x):
    return x**-1
def softmax(x, prevexpo) -> int:
    return 2.71828 ** x/prevexpo
def binarycrossentropyloss(x,y) -> int:
    return -(x * math.log(y) + (1 - x) * log(1 -y))
def alphanode(x, x0) -> bool:
    if(x > x0):
        return x
    if(x < x0):
        return False
def activationNode(array: [int], pointer: int, large: int) -> int:
    answer = alphanode(array[pointer], array[pointer - 1])
    if(answer != False):
        large = answer
        pointer+1
    else:
        pointer+1

def backpropgation(X, y, weights, learning_rate):
    z = sum(X[i] * weights[i] for i in range(len(X)))
    a = relu(z)
    hat = 1 / (1 + math.exp(-a)) # Gradient descent
    tadel = y - hat
    gradient = [delta * a * learning_rate for a in X]
    aweights = [weights[i] + gradients[i] for i in range(len(weights))]
    return aweights
class NerualNetwork:
    weights: [[int]]
    neurons: [[int]]
    bias: [[int]]
    layerposition: int
    inputs: [int]
    def forward(self):
        layerposition+=1
        b: int = sum(inputs * weights[layerposition]) + inputs * bias
        self.inputs = relu(b)
        forward()
def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    input_size = len(X_train[0])
    network = NerualNetwork(input_size)
    output = network.forward()
    print(f"Output after forward pass: {output}")
    weights_updated = backpropgation(X_train[0], y_train[0], network.weights[0], 0.01)
    print(f"Updated weights after backpropagation: {weights_updated}")

if __name__ == "__main__":
    main()




