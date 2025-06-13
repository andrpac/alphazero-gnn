class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below.
    """

    def __init__(self, game):
        pass

    def train(self, examples, gnn_examples=None):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value.
            gnn_examples: a list of GNN training examples (optional), where each example
                      is of form (board, std_pi, std_v, gnn_pi, gnn_v, v). This is used
                      when training with GNN sliding window approach.
        """
        pass

    def predict(self, board):
        """
        Standard prediction without GNN enhancement.
        
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board
            v: a float in [-1,1] that gives the value of the current board
        """
        pass
        
    def predict_with_gnn(self, board):
        """
        Prediction with GNN enhancement.
        
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board using GNN enhancement
            v: a float in [-1,1] that gives the value of the current board using GNN enhancement
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass