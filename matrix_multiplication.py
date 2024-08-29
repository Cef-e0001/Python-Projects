# Matrix multiplication from https://www.programiz.com/python-programming/examples/multiply-matrix

# Print matrix prettier
def printMatrix(mat):
    for row in mat:
        line = ""
        for val in row:
            line += str(val).rjust(6)+"\t"
        line = line[:-1]
        print(line)

# 3x3 matrix
X = [[12,7,3],
    [4 ,5,6],
    [7 ,8,9]]

# 3x4 matrix
Y = [[5,8,1,2],
    [6,7,3,0],
    [4,5,9,1]]

# result is 3x4
result = [
    [
        sum(a*b for a,b in zip(X_row,Y_col)) 
        for Y_col in zip(*Y)] 
        for X_row in X
    ]

''' 
    The output of this program is the same as above. 
    To understand the above code we must first know about 
    built-in function zip() and unpacking argument list 
    using * operator.

    The only concern is that it doesn't detect row-collumn mismatch.
    Well, it is what it is.
'''

printMatrix(result)