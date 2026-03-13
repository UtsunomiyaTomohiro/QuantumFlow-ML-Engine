import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

def objective(trial):
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Define search space
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return cross_val_score(clf, X, y, n_jobs=-1, cv=3).mean()

class QuantumFlowEngine:
    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        self.study = optuna.create_study(direction='maximize')

    def run_optimization(self):
        print(f"🚀 QuantumFlow: Starting Bayesian Optimization ({self.n_trials} trials)...")
        self.study.optimize(objective, n_trials=self.n_trials)
        print("\n✅ Optimization Complete.")
        print(f"🏆 Best Hyperparameters: {self.study.best_params}")
        print(f"📈 Best Validation Score: {self.study.best_value:.4f}")

if __name__ == "__main__":
    engine = QuantumFlowEngine()
    engine.run_optimization()