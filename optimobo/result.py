import matplotlib.pyplot as plt


class Res():
    """
    Class created to hold all the data and functions associated with the result of optimisation.
    """

    def __init__(self, pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, n_obj, n_init_samples):
        self.pf_approx = pf_approx
        self.pf_inputs = pf_inputs
        self.ysample = ysample
        self.Xsample = Xsample
        self.hypervolume_convergence = hypervolume_convergence
        self.n_obj = n_obj
        self.n_init_samples = n_init_samples

    def plot_pareto_front(self):
        """
        Using Matplotlib, this plots the pareto front for 2 and 3 objective problems.
        """
        if self.n_obj == 2:
            plt.scatter(self.ysample[5:,0], self.ysample[5:,1], color="red", label="Samples.")
            plt.scatter(self.pf_approx[:,0], self.pf_approx[:,1], color="green", label="PF approximation.")
            plt.scatter(self.ysample[0:self.n_init_samples,0], self.ysample[0:self.n_init_samples,1], color="blue", label="Initial samples.")
            plt.scatter(self.ysample[-1:-5:-1,0], self.ysample[-1:-5:-1,1], color="black", label="Last 5 samples.")
            plt.xlabel(r"$f_1(x)$")
            plt.ylabel(r"$f_2(x)$")
            plt.legend()
        
        elif self.n_obj == 3:
            fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
            ax1.scatter(self.ysample[5:,0], self.ysample[5:,1], self.ysample[5:,2], color="red", label="Samples.")
            ax1.scatter(self.ysample[0:self.n_init_samples,0], self.ysample[0:self.n_init_samples,1], self.ysample[0:self.n_init_samples,2], color="blue", label="Initial samples.")
            ax1.scatter(self.pf_approx[:,0], self.pf_approx[:,1], self.pf_approx[:,2], color="green", label="PF approximation.")
            ax1.scatter(self.ysample[-1:-5:-1,0], self.ysample[-1:-5:-1,1], self.ysample[-1:-5:-1,2], color="black", label="Last 5 samples.")
            ax2.scatter(self.pf_approx[:,0], self.pf_approx[:,1], self.pf_approx[:,2], color="green", label="PF approximation.")
            ax1.set_xlabel(r"$f_1(x)$")
            ax1.set_ylabel(r"$f_2(x)$")
            ax1.set_zlabel(r"$f_3(x)$")
            ax2.set_xlabel(r"$f_1(x)$")
            ax2.set_ylabel(r"$f_2(x)$")
            ax2.set_zlabel(r"$f_3(x)$")
            ax1.legend()


    def plot_hv_convergence(self):
        """
        So you can see how the hypervolume changes from iteration to iteration.
        """
        plt.plot(self.hypervolume_convergence)
    


class Constrained_Res(Res):

    """
    Changed to 
    """

    def __init__(self, y_infeasible, y_feasible, X_infeasible, X_feasible, pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, n_obj, n_init_samples):
        super().__init__(pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, n_obj, n_init_samples)
        self.X_infeasible = X_infeasible
        self.X_feasible = X_feasible
        self.y_feasible = y_feasible
        self.y_infeasible = y_infeasible

    def plot_pareto_front(self):
        """
        Using Matplotlib, this plots the pareto front for 2 and 3 objective problems.
        """
        if self.n_obj == 2:
            plt.scatter(self.y_feasible[5:,0], self.y_feasible[5:,1], color="c", label="Feasible.")
            plt.scatter(self.y_infeasible[5:,0], self.y_infeasible[5:,1], color="m", label="Infeasible.")
            plt.scatter(self.pf_approx[:,0], self.pf_approx[:,1], color="green", label="PF approximation.")
            plt.scatter(self.ysample[0:self.n_init_samples,0], self.ysample[0:self.n_init_samples,1], color="blue", label="Initial samples.")
            plt.scatter(self.ysample[-1:-5:-1,0], self.ysample[-1:-5:-1,1], color="black", label="Last 5 samples.")
            plt.xlabel(r"$f_1(x)$")
            plt.ylabel(r"$f_2(x)$")
            plt.legend()
        
        elif self.n_obj == 3:
            fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
            ax1.scatter(self.y_feasible[5:,0], self.y_feasible[5:,1], self.y_feasible[5:,2], color="c", label="Feasible.")
            ax1.scatter(self.y_infeasible[5:,0], self.y_infeasible[5:,1], self.y_infeasible[5:,2], color="m", label="Infeasible.")
            ax1.scatter(self.ysample[0:self.n_init_samples,0], self.ysample[0:self.n_init_samples,1], self.ysample[0:self.n_init_samples,2], color="blue", label="Initial samples.")
            ax1.scatter(self.pf_approx[:,0], self.pf_approx[:,1], self.pf_approx[:,2], color="green", label="PF approximation.")
            ax1.scatter(self.ysample[-1:-5:-1,0], self.ysample[-1:-5:-1,1], self.ysample[-1:-5:-1,2], color="black", label="Last 5 samples.")
            ax2.scatter(self.pf_approx[:,0], self.pf_approx[:,1], self.pf_approx[:,2], color="green", label="PF approximation.")
            ax1.set_xlabel(r"$f_1(x)$")
            ax1.set_ylabel(r"$f_2(x)$")
            ax1.set_zlabel(r"$f_3(x)$")
            ax2.set_xlabel(r"$f_1(x)$")
            ax2.set_ylabel(r"$f_2(x)$")
            ax2.set_zlabel(r"$f_3(x)$")
            ax1.legend()

    