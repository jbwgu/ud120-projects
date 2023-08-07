#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    predictions = list(predictions)
    ages = list(ages)
    net_worths = list(net_worths)
    errors = []
    for i in range(len(predictions)):
        errors.append(abs(predictions[i] - net_worths[i]))
    errors = list(errors)
    for i in range(int(len(predictions)*0.9)):
        index = errors.index(max(errors))
        errors.pop(index)
        cleaned_data.append((ages.pop(index), net_worths.pop(index), max(errors)))

    
    return cleaned_data

