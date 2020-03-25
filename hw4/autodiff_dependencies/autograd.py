TRACE_EVAL = False
TRACE_BP = False
from functions import *
import pdb

class Autograd(object):

    def __init__(self,xman):
        self.xman = xman

    def eval(self,opseq,valueDict):
        """ Evaluate the function defined by the operation sequence, where
        valueDict is a dict holding the values of any
        inputs/parameters that are needed (indexed by register name).
        """
        for (dstName,funName,inputNames) in opseq:
            if TRACE_EVAL: 
                print ('eval:',dstName,'=',funName,inputNames)
            inputValues = map(lambda a:valueDict[a] if a in valueDict else a.default, inputNames)
            if TRACE_EVAL:
                print ([(a,b.shape) for a,b in zip(inputNames, inputValues)])
            fun = EVAL_FUNS[funName] 
            result = fun(*inputValues)
            valueDict[dstName] = result
        return valueDict

    def bprop(self,opseq,valueDict,**deltaDict):
        """ For each intermediate register g used in computing the function f
        associated with the opseq, find df/dg.  Here valueDict is a
        dict holding the values of any inputs/parameters that are
        needed for the gradient (indexed by register name), as
        returned by eval.
        """
        for (dstName,funName,inputNames) in self.optimizeForBProp(opseq):
            delta = deltaDict[dstName]
            if TRACE_BP: 
                print ('bprop [',delta,']',dstName,'=',funName,inputNames)
            # values will be extended to include the next-level delta
            # and the output, and these will be passed as arguments
            values = [delta] + list(map(lambda a:valueDict[a], [dstName]+list(inputNames)))
            for i in range(len(list(inputNames))):
                if TRACE_BP: 
                    print(' -',dstName,'->',funName,'-> (...',inputNames[i],'...)')
                if TRACE_BP:
                    print([('delta', delta.shape),(inputNames[i], valueDict[inputNames[i]].shape)])

                result = (BP_FUNS[funName][i])(*values)
                # increment a running sum of all the delta's that are
                # pushed back to the i-th parameter, initializing the
                # zero if needed.
                self._incrementBy(deltaDict, inputNames[i], result)
        return deltaDict

    def _incrementBy(self, dict, key, inc):
        if key not in dict: dict[key] = inc
        else: dict[key] = dict[key] + inc

    def optimizeForBProp(self,opseq):
        """ Optimize an operation sequence for backprop.  Currently, reverse
        it and replace any occurence of "z=crossEnt(a,b), ...,
        a=softMax(c)" with with "z=crossEnt-softMax(c,b)"
        """
        opseq = list(reversed(opseq))
        opseq_new = [(o[0], o[1], list(o[2])) for o in opseq]

        # find where z = f(...) appears
        def find(dst=None,fun=None):
            def match(actual,target): return target==None or actual==target
            for k,(dstName,funName,inputNames) in enumerate(opseq):
                if match(dstName,dst) and match(funName,fun):
                    return k
            return -1
        # look for places to optimize
        crossEntOptimizations = []
        for k,(dstName,funName,inputNames) in enumerate(opseq_new):
            # look for z=crossEnt(softMax(p), y) where y is an input or param
            if funName=='crossEnt':
                (a,b) = list(inputNames); ka = find(dst=a); kb = find(dst=b)
                if ka>=0 and kb<0 and opseq[ka][1]=='softMax':
                    crossEntOptimizations.append((k,ka))
        # perform the optimization, by splicing out operation index ka
        # and replacing operation k with a single crossEnt-softMax
        # operation

        for (k,ka) in crossEntOptimizations:
            z = opseq[k][0]
            # pdb.set_trace()
            b = opseq_new[k][2][1]
            c = opseq_new[ka][2][0]
            opseq = opseq_new[:k] + [(z,'crossEnt-softMax',(c,b))] + opseq_new[k+1:ka]+opseq_new[ka+1:]
        return opseq
