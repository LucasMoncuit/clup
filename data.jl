# common part of the TP, define the criterion

function build_X(nVariables, nCassures, xMin, xMax)
    return xMin .+ randn(nVariables, nCassures) .* (xMax - xMin)
end

function build_bounds(nVariables)
    lb = -10.0 * ones(nVariables)
    ub =  10.0 * ones(nVariables)
    return lb, ub
end
