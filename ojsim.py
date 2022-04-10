from dataset import DataSet
from critic import Critic

class OJSimulator:
    def __init__(self, log_price_file = "./log_price.df", 
                        volume_file="./volume_usd.df", 
                        test_size=1440 * 60):
        self.dataset_ = DataSet(log_price_file, volume_file, test_size)
        self.critic_ = Critic()


    def submit(self, get_r_hat):
        log_pr, volu = self.dataset_.test_set
        self.critic_.submit(get_r_hat, log_pr, volu)

    
    @property
    def dataset(self):
        return self.dataset_

    
    @property
    def critic(self):
        return self.critic_


    @property
    def train(self):
        return self.dataset_.train_set

    
    @property
    def test(self):
        return self.dataset_.test_set


    @property
    def formulized_train(self):
        return self.dataset_.train_set_form


    
    