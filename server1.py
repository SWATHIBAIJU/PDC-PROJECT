import flwr as fl
import sys
import numpy as np
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):      
        if results:
            acc = [result['accuracy'] for result in results.values()]
            
            loss = [result['loss'] for result in results.values()]
            average_accuracy = np.mean(acc)
            loss = np.mean(loss)
            print(f"loss={loss}, acc={average_accuracy}")

# Create strategy and run server
def main():
 strategy = SaveModelStrategy()
 fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1000),
        strategy=strategy,
        
    )
if __name__ == "__main__":
    main()
