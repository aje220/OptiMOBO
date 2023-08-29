import numpy as np
import matplotlib.pyplot as plt
import util_functions
from scipy import stats



class EMO:


    def decompose_into_cells(self, data_points, ref_point):
        def wfg(pl, ref_point):
            """
            L. While et al. 
            10.1109/TEVC.2010.2077298
            Algorithm for calculating the hypervolume of a set of points. Assumes minimisation.
            
            Params:
                pl: set of points.
                ref_point: the coordinate from which to measure hypervolume, the reference point.
            
            """

            # return sum([exclhv(pl, k, ref_point) for k in range(len(pl))])
            return [exclhv(pl, k, ref_point) for k in range(len(pl))]


        def exclhv(pl, k, ref_point):

            limit_set = limitset(pl, k)
            ls_hv = wfg(util_functions.calc_pf(limit_set), ref_point)
            return limit_set
            
        def limitset(pl, k):
            result = []
            for j in range(len(pl)-k-1):
                aux = []
                for (p, q) in zip(pl[k], pl[j+k+1]):
                    res = min(p,q)
                    aux.append(res)
                result.append(aux)
            # result = [[max(p,q) for (p,q) in zip(pl[k], pl[j+k+1])] for j in range(len(pl)-k-1)]
            return result

        def inclhv(p, ref_point):

            return np.product([np.abs(p[j] - ref_point[j]) for j in range(2)])

        # Sorting the coords makes understanding whats going on easier
        sorted_coordinates = sorted(data_points, key=lambda coord: coord[0])

        # Get the limitsets of the pareto set of the points
        ss = np.asarray(wfg(util_functions.calc_pf(sorted_coordinates), ref_point))

        # The final limitset needs to be fixed to include the correct point, this is due to a limitation of the modified
        # wfg algorithm
        ss[-1] = [[ref_point[0], util_functions.calc_pf(sorted_coordinates)[-1][0]]]

        # We get the upper bounds of each cell
        upperlower = [[sorted_coordinates[i], ss[i][0]] for i, _ in enumerate(ss)]

        # Now in this loop we include the other coordinates for each cell
        new = upperlower
        bbbb = []
        asdff = []
        for i in upperlower:
            asdf = []
            for j in i:
                asdf.append([1,j[1]])
                # print([1,j[1]])
                bbbb.append([1,j[1]])
                # print(j)
            asdff.append(asdf)

        # stack the coordinates together
        final = np.hstack((upperlower, asdff))
        # import pdb; pdb.set_trace()
        # We need to fix the final cell again.
        final[-1][-1] = ref_point
        return final
    
    def hypervolume_improvement(self, query_point, P, ref_point):
        """
        query_point: objective vector to query:
        P pareto set
        Returns the improvment in hypervolume from the inclusion of the query point
        into the set of objective values.
        """
        before = util_functions.wfg(P, ref_point)
    
        aggregated = np.vstack((P,query_point))
        after = util_functions.wfg(aggregated, ref_point)
        improvement = after - before 
        import pdb; pdb.set_trace()
        if improvement > 0:
            return improvement
        else:
            return 0

    def hypervolume_based_PoI(self, cells):



        vol

        return
        







if __name__ == "__main__":



    arr = np.asarray([[3, 7],
       [6, 2],
       [4, 5],
       [7, 3],
       [2, 8],
       [3, 3]])

    plt.scatter(arr[:,0], arr[:,1])
    pf = util_functions.calc_pf(arr)
    plt.scatter(pf[:,0], pf[:,1])
    plt.grid(True)
    plt.show()

    print(util_functions.wfg(arr, [8,9]))
    print(util_functions.exclhv(arr, 5, [8,9]))
    optimiser = EMO()
    objectives2 = np.asarray([
        [3, 7],
        [6, 2],
        [4, 5],
        [7, 3],
        [2, 8]
    ])

    xxx = optimiser.hypervolume_improvement([3,3], objectives2, [8,9])

    print(xxx)

    





# objectives2 = np.asarray([
#     [2,5],
#     [3,4],
#     [4,3],
#     [2,7]
# ])



