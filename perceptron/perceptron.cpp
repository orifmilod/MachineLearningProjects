#include <iostream>
#include <vector>

using namespace std;

class Perceptron {
public:
  Perceptron(float _eta, int _epochs) {
    epochs = _epochs;
    eta = _eta;
  }
  float netInput(vector<float> X);
  int predict(vector<float> X);
  void fit(vector<vector<float>> X, vector<float> y);

private:
  int epochs;
  float eta;
  vector<float> weights;
};

void Perceptron::fit(vector<vector<float>> X, vector<float> y) {
  weights = vector<float>(X[0].size() + 1, 0);

  for (int i = 0; i < epochs; i++) // Iterating through each epoch
  {
    for (int j = 0; j < X.size(); j++) {
      float update = eta * (y[j] - predict(X[j])); // change for the weights

      for (int w = 1; w < weights.size(); w++) {
        // Update each weight by the update * the training sample
        weights[w] += update * X[j][w - 1];
      }
      weights[0] =
          update; // We update the Bias term and setting it equal to the update
    }
  }
}
int Perceptron::predict(vector<float> X) {
  return netInput(X) > 0 ? 1 : -1; // Step Function
}

float Perceptron::netInput(vector<float> X) {
  float probabilities = weights[0]; // adding the perceptron first

  for (int i = 0; i < X.size(); i++) {
    probabilities += X[i] * weights[i + 1];
  }
  return probabilities;
}

int main() { return 0; }
