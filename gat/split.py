def completeSplit(L):
    for i in range(len(L)):
        if L[i][len(L[i]) - 1] == ",":
            beginL = L[:i - 1]
            endL = L[i + 1:]
            L[i] = L[i][:len(L[i]) - 1]
            L = beginL + [L[i], ','] + endL
        elif L[i][len(L[i]) - 1] == ";":
            beginL = L[:i - 1]
            endL = L[i + 1:]
            L[i] = L[i][:len(L[i]) - 1]
            L = beginL + [L[i], ';'] + endL
        elif L[i][len(L[i]) - 1] == ".":
            beginL = L[:i - 1]
            endL = L[i + 1:]
            L[i] = L[i][:len(L[i]) - 1]
            L = beginL + [L[i], '.'] + endL
    return L
