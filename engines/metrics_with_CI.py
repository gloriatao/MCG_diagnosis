from __future__ import print_function, division
from math import sqrt

from scipy.special import ndtri

def _proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.
    
    Follows notation described on pages 46--47 of [1]. 
    
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    """
    
    A = 2*r + z**2
    B = z*sqrt(z**2 + 4*r*(1 - r/n))
    C = 2*(n + z**2)
    return (round((A-B)/C* 100, 1), round((A+B)/C* 100, 1))

def metrics_CI(TP, FP, FN, TN, alpha=0.95):
    """Compute confidence intervals for sensitivity and specificity using Wilson's method. 
    
    This method does not rely on a normal approximation and results in accurate 
    confidence intervals even for small sample sizes.
    
    Parameters
    ----------
    TP : int
        Number of true positives
    FP : int 
        Number of false positives
    FN : int
        Number of false negatives
    TN : int
        Number of true negatives
    alpha : float, optional
        Desired confidence. Defaults to 0.95, which yields a 95% confidence interval. 
    
    Returns
    -------
    sensitivity_point_estimate : float
        Numerical estimate of the test sensitivity
    specificity_point_estimate : float
        Numerical estimate of the test specificity
    sensitivity_confidence_interval : Tuple (float, float)
        Lower and upper bounds on the alpha confidence interval for sensitivity
    specificity_confidence_interval
        Lower and upper bounds on the alpha confidence interval for specificity 
        
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    [2] E. B. Wilson, Probable inference, the law of succession, and statistical inference,
    J Am Stat Assoc 22:209-12, 1927. 
    """
    
    # 
    z = -ndtri((1.0-alpha)/2)
    
    # Compute sensitivity using method described in [1]
    sensitivity_point_estimate = TP/(TP + FN) * 100
    sensitivity_confidence_interval = _proportion_confidence_interval(TP, TP + FN, z)
    
    # Compute specificity using method described in [1]
    specificity_point_estimate = TN/(TN + FP)* 100
    specificity_confidence_interval = _proportion_confidence_interval(TN, TN + FP, z)
    
    NPV = TN/(TN+FN)* 100
    NPV_confidence_interval = _proportion_confidence_interval(TN, TN+FN, z)
    
    PPV = TP/(TP+FP) * 100
    PPV_confidence_interval = _proportion_confidence_interval(TP, TP+FP, z)    
    
    Accuracy = (TP+TN)/(TP+TN+FP+FN) * 100
    Accuracy_confidence_interval = _proportion_confidence_interval(TP+TN, TP+TN+FP+FN, z) 
    
    F1 = 2*TP/(2*TP+FP+FN)* 100
    F1_confidence_interval = _proportion_confidence_interval(2*TP, 2*TP+FP+FN, z) 
    
    return round(sensitivity_point_estimate,1), round(specificity_point_estimate,1), sensitivity_confidence_interval, specificity_confidence_interval,\
        round(NPV,1), NPV_confidence_interval, round(PPV,1), PPV_confidence_interval, round(Accuracy,1), Accuracy_confidence_interval, round(F1,1), F1_confidence_interval


# for a in [0.95]:
#     sensitivity_point_estimate, specificity_point_estimate, \
#         sensitivity_confidence_interval, specificity_confidence_interval \
#         = metrics_CI(5, 3, 1, 117, alpha=a)
#     print("Sensitivity: %f, Specificity: %f" %(sensitivity_point_estimate, specificity_point_estimate))
#     print("alpha = %f CI for sensitivity:"%a, sensitivity_confidence_interval)
#     print("alpha = %f CI for specificity:"%a, specificity_confidence_interval)
#     print("")