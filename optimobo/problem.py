from abc import abstractmethod

import numpy as np

import pymoo.gradient.toolbox as anp
from pymoo.util.cache import Cache
from pymoo.util.misc import at_least_2d_array

try:
    import ray
except ImportError:
    ray = None


class ElementwiseEvaluationFunction:

    def __init__(self, problem, args, kwargs) -> None:
        super().__init__()
        self.problem = problem
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        out = dict()
        self.problem._evaluate(x, out, *self.args, **self.kwargs)
        return out

class ElementwiseEvaluationFunctionConstraint:
    
    def __init__(self, problem, args, kwargs) -> None:
        super().__init__()
        self.problem = problem
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        out = dict()
        self.problem._evaluate_constraints(x, out, *self.args, **self.kwargs)
        return out


class LoopedElementwiseEvaluation:

    def __call__(self, f, X):
        return [f(x) for x in X]


class StarmapParallelization:

    def __init__(self, starmap) -> None:
        super().__init__()
        self.starmap = starmap

    def __call__(self, f, X):
        return list(self.starmap(f, [[x] for x in X]))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("starmap", None)
        return state


class DaskParallelization:

    def __init__(self, client) -> None:
        super().__init__()
        self.client = client

    def __call__(self, f, X):
        jobs = [self.client.submit(f, x) for x in X]
        return [job.result() for job in jobs]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)
        return state


class JoblibParallelization:

    def __init__(self, aJoblibParallel,  aJoblibDelayed, *args, **kwargs) -> None:
        super().__init__()
        self.parallel = aJoblibParallel 
        self.delayed =  aJoblibDelayed 

    def __call__(self, f, X):
        return self.parallel(self.delayed(f)(x) for x in X)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("parallel", None)
        state.pop("delayed", None)
        return state


class RayParallelization:
    """Use Ray as backend to parallelize problem evaluation.
    
    Ray is an open-source unified framework for scaling AI and Python applicaitons.
    Read more here: https://docs.ray.io.
    
    You will need to install Ray to use this.
    """
    def __init__(self, job_resources: dict = {'num_cpus': 1}) -> None:
        """
        Parameters
        ----------
        job_resources: A resource in Ray is a key-value pair where the key denotes a 
            resource name and the value is a float quantity. Ray has native support for CPU,
            GPU, and memory resource types; `'num_cpus'`, `'num_gpus'`, and `'memory'`.
            Read more here: 
            https://docs.ray.io/en/latest/ray-core/scheduling/resources.html.
        """
        assert ray is not None, (
            "Ray must be installed! "
            "You can install Ray with the command: "
            '`pip install -U "ray[default]"`'
        )
        super().__init__()
        self.job_resources = job_resources

    def __call__(self, f, X):
        runnable = ray.remote(f.__call__.__func__)
        runnable = runnable.options(**self.job_resources)
        futures = [runnable.remote(f, x) for x in X]
        return ray.get(futures)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state


class Problem:
    def __init__(self,
                 n_var=-1,
                 n_obj=1,
                 n_ieq_constr=0,
                 n_eq_constr=0,
                 xl=None,
                 xu=None,
                 vtype=None,
                 vars=None,
                 elementwise=False,
                 elementwise_func=ElementwiseEvaluationFunction,
                 elementwise_func_constr=ElementwiseEvaluationFunctionConstraint,
                 elementwise_runner=LoopedElementwiseEvaluation(),
                 requires_kwargs=False,
                 replace_nan_values_by=None,
                 exclude_from_serialization=None,
                 callback=None,
                 strict=True,
                 **kwargs):

        """

        Parameters
        ----------
        n_var : int
            Number of Variables

        n_obj : int
            Number of Objectives

        n_ieq_constr : int
            Number of Inequality Constraints

        n_eq_constr : int
            Number of Equality Constraints

        xl : np.array, float, int
            Lower bounds for the variables. if integer all lower bounds are equal.

        xu : np.array, float, int
            Upper bounds for the variable. if integer all upper bounds are equal.

        vtype : type
            The variable type. So far, just used as a type hint.

        """


        # SO this is all initalisation

        # number of variable
        self.n_var = n_var

        # number of objectives
        self.n_obj = n_obj

        # number of inequality constraints
        self.n_ieq_constr = n_ieq_constr if "n_constr" not in kwargs else max(n_ieq_constr, kwargs["n_constr"])

        # number of equality constraints
        self.n_eq_constr = n_eq_constr

        # type of the variable to be evaluated
        self.data = dict(**kwargs)

        # the lower bounds, make sure it is a numpy array with the length of n_var
        self.xl, self.xu = xl, xu

        # a callback function to be called after every evaluation
        self.callback = callback

        # if the variables are provided in their explicit form
        if vars is not None:
            self.vars = vars
            self.n_var = len(vars)

            if self.xl is None:
                self.xl = {name: var.lb if hasattr(var, "lb") else None for name, var in vars.items()}
            if self.xu is None:
                self.xu = {name: var.ub if hasattr(var, "ub") else None for name, var in vars.items()}

        # the variable type (only as a type hint at this point)
        self.vtype = vtype

        # the functions used if elementwise is enabled
        self.elementwise = elementwise
        self.elementwise_func = elementwise_func
        self.elementwise_func_constr = elementwise_func_constr
        self.elementwise_runner = elementwise_runner
        

        # whether evaluation requires kwargs (passing them can cause overhead in parallelization)
        self.requires_kwargs = requires_kwargs

        # whether the shapes are checked strictly
        self.strict = strict

        # if it is a problem with an actual number of variables - make sure xl and xu are numpy arrays
        if n_var > 0:

            if self.xl is not None:
                if not isinstance(self.xl, np.ndarray):
                    self.xl = np.ones(n_var) * xl
                self.xl = self.xl.astype(float)

            if self.xu is not None:
                if not isinstance(self.xu, np.ndarray):
                    self.xu = np.ones(n_var) * xu
                self.xu = self.xu.astype(float)

        # this defines if NaN values should be replaced or not
        self.replace_nan_values_by = replace_nan_values_by

        # attribute which are excluded from being serialized
        self.exclude_from_serialization = exclude_from_serialization

    def evaluate(self,
                 X,
                 *args,
                 return_values_of=None,
                 return_as_dictionary=False,
                 **kwargs):

        # if the problem does not require any kwargs they are re-initialized
        if not self.requires_kwargs:
            kwargs = dict()

        # it defines what is actually expected to return, 
        if return_values_of is None:
            return_values_of = ["F"]
            # if self.n_ieq_constr > 0:
            #     return_values_of.append("G")
            # if self.n_eq_constr > 0:
            #     return_values_of.append("H")

        # make sure the array is at least 2d. store if reshaping was necessary
        if isinstance(X, np.ndarray) and X.dtype != object:
            X, only_single_value = at_least_2d_array(X, extend_as="row", return_if_reshaped=True)
            assert X.shape[1] == self.n_var, f'Input dimension {X.shape[1]} are not equal to n_var {self.n_var}!'
        else:
            only_single_value = not (isinstance(X, list) or isinstance(X, np.ndarray))

        # this is where the actual evaluation takes place
        # calls do
        # return_values_of tells the evaluator what to return.
        _out = self.do(X, return_values_of, *args, **kwargs)

        out = {}
        # for key and value in out
        for k, v in _out.items():

            # copy it to a numpy array (it might be one of jax at this point)
            v = np.array(v)

            # in case the input had only one dimension, then remove always the first dimension from each output
            if only_single_value:
                v = v[0]

            # if the NaN values should be replaced
            if self.replace_nan_values_by is not None:
                v[np.isnan(v)] = self.replace_nan_values_by

            # set out to the values of _out
            try:
                out[k] = v.astype(np.float64)
            except:
                out[k] = v

        # if callback function is defined, call it
        if self.callback is not None:
            self.callback(X, out)

        # now depending on what should be returned prepare the output
        if return_as_dictionary:
            return out

        # if there is only one value to return, return valu
        if len(return_values_of) == 1:
            return out[return_values_of[0]]
        else:
            # so return values of has  a seperation between objective values and constraints.
            return tuple([out[e] for e in return_values_of])

    def do(self, X, return_values_of, *args, **kwargs):

        # this is called to do the actual evaluation

        # create an empty dictionary
        # return_values_of is an array telling this dict what keys to create,
        # it can just do "F", but if you include constraints this dict will have keys of "G" and "H"
        out = {name: None for name in return_values_of}

        # do the function evaluation
        if self.elementwise:
            self._evaluate_elementwise(X, out, *args, **kwargs)
        else:
            # if vectorised, this goes on to call the abstract method that the user implements.
            self._evaluate_vectorized(X, out, *args, **kwargs)

        # finally format the output dictionary
        out = self._format_dict(out, len(X), return_values_of)

        return out

    def _evaluate_vectorized(self, X, out, *args, **kwargs):
        # call the user implemented abstract method
        self._evaluate(X, out, *args, **kwargs)

    def _evaluate_elementwise(self, X, out, *args, **kwargs):

        # create the function that evaluates a single individual
        f = self.elementwise_func(self, args, kwargs)

        # execute the runner
        elems = self.elementwise_runner(f, X)

        # for each evaluation call
        for elem in elems:

            # for each key stored for this evaluation
            for k, v in elem.items():

                # if the element does not exist in out yet -> create it
                if out.get(k, None) is None:
                    out[k] = []

                out[k].append(v)

        # convert to arrays (the none check is important because otherwise an empty array is initialized)
        for k in out:
            if out[k] is not None:
                out[k] = anp.array(out[k])

    def _format_dict(self, out, N, return_values_of):

        # get the default output shape for the default values
        shape = default_shape(self, N)

        # finally the array to be returned
        ret = {}

        # for all values that have been set in the user implemented function
        for name, v in out.items():

            # only if they have truly been set
            if v is not None:

                # if there is a shape to be expected
                if name in shape:

                    if isinstance(v, list):
                        v = anp.column_stack(v)

                    try:
                        v = v.reshape(shape[name])
                    except Exception as e:
                        raise Exception(
                            f"Problem Error: {name} can not be set, expected shape {shape[name]} but provided {v.shape}",
                            e)

                ret[name] = v

        # if some values that are necessary have not been set
        for name in return_values_of:
            if name not in ret:
                s = shape.get(name, N)
                ret[name] = np.full(s, np.inf)

        return ret
#######################################################################################
    
    def evaluate_constraints(self,
                 X,
                 *args,
                 return_values_of=None,
                 return_as_dictionary=False,
                 **kwargs):
        """
        This is just for evaling constraints, its seperate so we can evaluate them without evaluating objective values
        """

        # if the problem does not require any kwargs they are re-initialized
        if not self.requires_kwargs:
            kwargs = dict()

        # it defines what is actually expected to return, 
        if return_values_of is None:
            # return_values_of = ["F"]
            return_values_of = []
            if self.n_ieq_constr > 0:
                return_values_of.append("G")
            if self.n_eq_constr > 0:
                return_values_of.append("H")

        # make sure the array is at least 2d. store if reshaping was necessary
        if isinstance(X, np.ndarray) and X.dtype != object:
            X, only_single_value = at_least_2d_array(X, extend_as="row", return_if_reshaped=True)
            assert X.shape[1] == self.n_var, f'Input dimension {X.shape[1]} are not equal to n_var {self.n_var}!'
        else:
            only_single_value = not (isinstance(X, list) or isinstance(X, np.ndarray))

        # this is where the actual evaluation takes place
        # calls do
        # return_values_of tells the evaluator what to return.
        _out = self.do_constraints(X, return_values_of, *args, **kwargs)

        out = {}
        # for key and value in out
        for k, v in _out.items():

            # copy it to a numpy array (it might be one of jax at this point)
            v = np.array(v)

            # in case the input had only one dimension, then remove always the first dimension from each output
            if only_single_value:
                v = v[0]

            # if the NaN values should be replaced
            if self.replace_nan_values_by is not None:
                v[np.isnan(v)] = self.replace_nan_values_by

            # set out to the values of _out
            try:
                out[k] = v.astype(np.float64)
            except:
                out[k] = v

        # if callback function is defined, call it
        if self.callback is not None:
            self.callback(X, out)

        # now depending on what should be returned prepare the output
        if return_as_dictionary:
            return out

        # if there is only one value to return, return valu
        if len(return_values_of) == 1:
            return out[return_values_of[0]]
        else:
            # so return values of has  a seperation between objective values and constraints.
            return tuple([out[e] for e in return_values_of])

    def do_constraints(self, X, return_values_of, *args, **kwargs):
        
        # this is called to do the actual evaluation

        # create an empty dictionary
        # return_values_of is an array telling this dict what keys to create,
        # it can just do "F", but if you include constraints this dict will have keys of "G" and "H"
        out = {name: None for name in return_values_of}

        # do the function evaluation
        if self.elementwise:
            self._evaluate_elementwise_constraints(X, out, *args, **kwargs)
        else:
            # if vectorised, this goes on to call the abstract method that the user implements.
            self._evaluate_vectorized_constraints(X, out, *args, **kwargs)

        # finally format the output dictionary
        out = self._format_dict(out, len(X), return_values_of)

        return out

    def _evaluate_vectorized_constraints(self, X, out, *args, **kwargs):
        # call the user implemented abstract method
        self._evaluate_constraints(X, out, *args, **kwargs)

    def _evaluate_elementwise_constraints(self, X, out, *args, **kwargs):

        # create the function that evaluates a single individual
        f = self.elementwise_func_constr(self, args, kwargs)

        # execute the runner
        elems = self.elementwise_runner(f, X)

        # for each evaluation call
        for elem in elems:

            # for each key stored for this evaluation
            for k, v in elem.items():

                # if the element does not exist in out yet -> create it
                if out.get(k, None) is None:
                    out[k] = []

                out[k].append(v)

        # convert to arrays (the none check is important because otherwise an empty array is initialized)
        for k in out:
            if out[k] is not None:
                out[k] = anp.array(out[k])

    def _format_dict(self, out, N, return_values_of):

        # get the default output shape for the default values
        shape = default_shape(self, N)

        # finally the array to be returned
        ret = {}

        # for all values that have been set in the user implemented function
        for name, v in out.items():

            # only if they have truly been set
            if v is not None:

                # if there is a shape to be expected
                if name in shape:

                    if isinstance(v, list):
                        v = anp.column_stack(v)

                    try:
                        v = v.reshape(shape[name])
                    except Exception as e:
                        raise Exception(
                            f"Problem Error: {name} can not be set, expected shape {shape[name]} but provided {v.shape}",
                            e)

                ret[name] = v

        # if some values that are necessary have not been set
        for name in return_values_of:
            if name not in ret:
                s = shape.get(name, N)
                ret[name] = np.full(s, np.inf)

        return ret
#####################################################################################
    @Cache
    def nadir_point(self, *args, **kwargs):
        pf = self.pareto_front(*args, **kwargs)
        if pf is not None:
            return np.max(pf, axis=0)

    @Cache
    def ideal_point(self, *args, **kwargs):
        pf = self.pareto_front(*args, **kwargs)
        if pf is not None:
            return np.min(pf, axis=0)

    @Cache
    def pareto_front(self, *args, **kwargs):
        pf = self._calc_pareto_front(*args, **kwargs)
        pf = at_least_2d_array(pf, extend_as='r')
        if pf is not None and pf.shape[1] == 2:
            pf = pf[np.argsort(pf[:, 0])]
        return pf

    @Cache
    def pareto_set(self, *args, **kwargs):
        ps = self._calc_pareto_set(*args, **kwargs)
        ps = at_least_2d_array(ps, extend_as='r')
        return ps

    @property
    def n_constr(self):
        return self.n_ieq_constr + self.n_eq_constr

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):
        pass

    @abstractmethod
    def _evaluate_constraints(self, x, out, *args, **kwargs):
        pass

    def has_bounds(self):
        return self.xl is not None and self.xu is not None

    def has_constraints(self):
        return self.n_constr > 0

    def bounds(self):
        return self.xl, self.xu

    def name(self):
        return self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        pass

    def _calc_pareto_set(self, *args, **kwargs):
        pass

    def __str__(self):
        s = "# name: %s\n" % self.name()
        s += "# n_var: %s\n" % self.n_var
        s += "# n_obj: %s\n" % self.n_obj
        s += "# n_ieq_constr: %s\n" % self.n_ieq_constr
        s += "# n_eq_constr: %s\n" % self.n_eq_constr
        return s

    def __getstate__(self):
        if self.exclude_from_serialization is not None:
            state = self.__dict__.copy()

            # exclude objects which should not be stored
            for key in self.exclude_from_serialization:
                state[key] = None

            return state
        else:
            return self.__dict__


class ElementwiseProblem(Problem):

    def __init__(self, elementwise=True, **kwargs):
        super().__init__(elementwise=elementwise, **kwargs)


def default_shape(problem, n):
    n_var = problem.n_var
    DEFAULTS = dict(
        F=(n, problem.n_obj),
        G=(n, problem.n_ieq_constr),
        H=(n, problem.n_eq_constr),
        dF=(n, problem.n_obj, n_var),
        dG=(n, problem.n_ieq_constr, n_var),
        dH=(n, problem.n_eq_constr, n_var),
    )
    return DEFAULTS


if __name__ == "__main__":

    class MyProblem(Problem):
        
        def __init__(self):
            super().__init__(n_var=2, n_obj=2, n_ieq_constr=2, vtype=float)
            self.xl = np.zeros(self.n_var)
            self.xu = np.array([5.0, 3.0])

        def _evaluate(self, x, out, *args, **kwargs):
            print("EVAL_CALLED")
            f1 = 4 * x[:, 0] ** 2 + 4 * x[:, 1] ** 2
            f2 = (x[:, 0] - 5) ** 2 + (x[:, 1] - 5) ** 2
            # out["F"] = anp.column_stack([f1, f2])
            out["F"] = [f1, f2]



        def _evaluate_constraints(self, x, out, *args, **kwargs):
            print("EVAL_CONSTR")
            g1 = (1 / 25) * ((x[:, 0] - 5) ** 2 + x[:, 1] ** 2 - 25)
            g2 = -1 / 7.7 * ((x[:, 0] - 8) ** 2 + (x[:, 1] + 3) ** 2 - 7.7)
            # out["G"] = anp.column_stack([g1, g2])
            out["G"] = [g1, g2]

        
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot
    problem = get_problem("bnh")
    inp = [1,1]
    res1 = problem.evaluate(np.asarray(inp))
    # res2 = problem.evaluate(np.asarray([2,2]), return_values_of="G")
    print("true: ")
    print(res1)
    # print(res2)
    print("\n")

    myprob = MyProblem()
    res1 = myprob.evaluate(np.asarray(inp))
    print(res1)
    res2 = myprob.evaluate_constraints(np.asarray(inp))
    print(res2)


    class OSY(Problem):
        def __init__(self):
            super().__init__(n_var=6, n_obj=2, n_ieq_constr=6, vtype=float)
            self.xl = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            self.xu = np.array([10.0, 10.0, 5.0, 6.0, 5.0, 10.0])

        def _evaluate(self, x, out, *args, **kwargs):
            print("EVAL_CALL")
            f1 = - (25 * (x[:, 0] - 2) ** 2 + (x[:, 1] - 2) ** 2 + (x[:, 2] - 1) ** 2 + (x[:, 3] - 4) ** 2 + (
                    x[:, 4] - 1) ** 2)
            f2 = anp.sum(anp.square(x), axis=1)

            out["F"] = anp.column_stack([f1, f2])

        
        def _evaluate_constraints(self, x, out, *args, **kwargs):
            print("CONSTR_CALLED")
            g1 = (x[:, 0] + x[:, 1] - 2.0) / 2.0
            g2 = (6.0 - x[:, 0] - x[:, 1]) / 6.0
            g3 = (2.0 - x[:, 1] + x[:, 0]) / 2.0
            g4 = (2.0 - x[:, 0] + 3.0 * x[:, 1]) / 2.0
            g5 = (4.0 - (x[:, 2] - 3.0) ** 2 - x[:, 3]) / 4.0
            g6 = ((x[:, 4] - 3.0) ** 2 + x[:, 5] - 4.0) / 4.0

            out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6])
            out["G"] = - out["G"]
            

        def _calc_pareto_front(self):
            return Remote.get_instance().load("pymoo", "pf", "osy.pf")


    problem = get_problem("OSY")
    inp = [5,5,3,3,3,5]
    res1 = problem.evaluate(np.asarray(inp))
    # res2 = problem.evaluate(np.asarray([2,2]), return_values_of="G")
    print("\ntrue: ")
    print(res1)
    # print(res2)
    print("\n")

    myprob = OSY()
    res1 = myprob.evaluate(np.asarray(inp))
    print(res1)
    res2 = myprob.evaluate_constraints(np.asarray(inp))
    print(res2)



    class Truss2D(Problem):
    
        def __init__(self):
            super().__init__(n_var=3, n_obj=2, n_ieq_constr=1, vtype=float)

            self.Amax = 0.01
            self.Smax = 1e5

            self.xl = np.array([0.0, 0.0, 1.0])
            self.xu = np.array([self.Amax, self.Amax, 3.0])

        def _evaluate(self, x, out, *args, **kwargs):
            print("EVAL CALLED")

            # variable names for convenient access
            x1 = x[:, 0]
            x2 = x[:, 1]
            y = x[:, 2]

            # first objectives
            f1 = x1 * anp.sqrt(16 + anp.square(y)) + x2 * anp.sqrt((1 + anp.square(y)))

            # measure which are needed for the second objective
            sigma_ac = 20 * anp.sqrt(16 + anp.square(y)) / (y * x1)
            sigma_bc = 80 * anp.sqrt(1 + anp.square(y)) / (y * x2)

            # take the max
            f2 = anp.max(anp.column_stack((sigma_ac, sigma_bc)), axis=1)

            out["F"] = anp.column_stack([f1, f2])

        
        def _evaluate_constraints(self, x, out, *args, **kwargs):
            print("CONSTR_CALLED")
            x1 = x[:, 0]
            x2 = x[:, 1]
            y = x[:, 2]


            # measure which are needed for the second objective
            sigma_ac = 20 * anp.sqrt(16 + anp.square(y)) / (y * x1)
            sigma_bc = 80 * anp.sqrt(1 + anp.square(y)) / (y * x2)

            # take the max
            f2 = anp.max(anp.column_stack((sigma_ac, sigma_bc)), axis=1)

            # define a constraint
            g1 = f2 - self.Smax
            out["G"] = g1


    problem = get_problem("Truss2D")
    inp = [5,5,3]
    res1 = problem.evaluate(np.asarray(inp))
    # res2 = problem.evaluate(np.asarray([2,2]), return_values_of="G")
    print("\ntrue Truss: ")
    print(res1)
    # print(res2)
    print("\n")

    myprob = Truss2D()
    res1 = myprob.evaluate(np.asarray(inp))
    print(res1)
    res2 = myprob.evaluate_constraints(np.asarray(inp))
    print(res2)



    class MODActMine(ElementwiseProblem):
        """Multi-Objective Design of Actuators

        MODAct is a framework for real-world constrained multi-objective optimization.
        Refer to the python package https://github.com/epfl-lamd/modact from requirements.

        Best-known Pareto fronts must be downloaded from here: https://doi.org/10.5281/zenodo.3824302

        Parameters
        ----------

        function: str or modact.problems
            The name of the benchmark problem to use either as a string or the
            problem object instance. Example values: cs1, cs3, ct2, ct4, cts3

        References:
        ----------
        C. Picard and J. Schiffmann, “Realistic Constrained Multi-Objective Optimization Benchmark Problems from Design,”
        IEEE Transactions on Evolutionary Computation, pp. 1–1, 2020.
        """

        def __init__(self, function, pf=None, **kwargs):

            self.function = function
            self.pf = pf

            try:
                import modact.problems as pb
            except:
                raise Exception("Please install the modact library: https://github.com/epfl-lamd/modact")

            if isinstance(function, pb.Problem):
                self.fct = function
            else:
                self.fct = pb.get_problem(function)

            lb, ub = self.fct.bounds()
            n_var = len(lb)
            n_obj = len(self.fct.weights)
            n_ieq_constr = len(self.fct.c_weights)
            xl = lb
            xu = ub

            self.weights = np.array(self.fct.weights)
            self.c_weights = np.array(self.fct.c_weights)

            super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu, vtype=float, **kwargs)

        def _evaluate(self, x, out, *args, **kwargs):
            print("EVAL_MODACT")
            f, _ = self.fct(x)
            out["F"] = np.array(f) * -1 * self.weights

        def _evaluate_constraints(self, x, out, *args, **kwargs):
            print("CONSTR_MODACT")
            _, g = self.fct(x)
            out["G"] = np.array(g) * self.c_weights


        def _calc_pareto_front(self, *args, **kwargs):
            # allows to provide a custom pf - because of the size of files published by the author
            if self.pf is None:
                pf = Remote.get_instance().load("pymoo", "pf", "MODACT", f"{self.function}.pf")
                # pf = pf * [1, -1]
                pf = pf * self.weights * -1
                return pf
            else:
                return self.pf
    
    from pymoo.problems.multi import MODAct
    problem = MODAct("cs1")
    inp = [  0.0,  0.3, 9.0, 30.0, 0.3,
   5.0, -20.0, -3.14159265, 9.0, 30.0,
   0.3, 5.0, -20.0, -3.14159265,   9.0,
  30.0, 0.3, 5.0, -20.0, -3.14159265]
    # inp = np.asarray([0.5]*20)


    res1 = problem.evaluate(np.asarray(inp))
    # res2 = problem.evaluate(np.asarray([2,2]), return_values_of="G")
    print("\ntrue Modact: ")
    print(res1)
    # print(res2)


    inp = [  0.0,  0.3, 9.0, 30.0, 0.3,
   5.0, -20.0, -3.14159265, 9.0, 30.0,
   0.3, 5.0, -20.0, -3.14159265,   9.0,
  30.0, 0.3, 5.0, -20.0, -3.14159265]
    print("\n")

    myprob = MODActMine("cs1")
    res1 = myprob.evaluate(np.asarray(inp))
    print(res1)
    res2 = myprob.evaluate_constraints(np.asarray(inp))
    print(res2)


    class ZDT(Problem):
    
        def __init__(self, n_var=5, **kwargs):
            super().__init__(n_var=n_var, n_obj=2, xl=0, xu=1, vtype=float, **kwargs)


    class ZDT1(ZDT):

        def _calc_pareto_front(self, n_pareto_points=100):
            x = np.linspace(0, 1, n_pareto_points)
            return np.array([x, 1 - np.sqrt(x)]).T

        def _evaluate(self, x, out, *args, **kwargs):
            print(x)
            f1 = x[:, 0]
            g = 1 + 9.0 / (self.n_var - 1) * anp.sum(x[:, 1:], axis=1)
            f2 = g * (1 - anp.power((f1 / g), 0.5))

            out["F"] = anp.column_stack([f1, f2])

    
    problem = get_problem("zdt1", n_var=5)
    inp = [0.5]*5
    res1 = problem.evaluate(np.asarray(inp))
    # res2 = problem.evaluate(np.asarray([2,2]), return_values_of="G")
    print("\ntrue zdt1: ")
    print(res1)
    # print(res2)
    print("\n")

    myprob = ZDT1()
    res1 = myprob.evaluate(np.asarray(inp))
    print(res1)
    res2 = myprob.evaluate_constraints(np.asarray(inp))
    print(res2)