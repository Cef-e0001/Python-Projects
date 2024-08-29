''' Simple Neural Network 
# Todo
[] learning algorithm: rebalance network.
    [] Inverse Matrix
[o] evaluation: make output.
    [o] Matrix multiplication
[] data representation: taking inputs.
[] network description: saving result.
[] varies hidden layer: customizer.
'''

'''create filled matrix'''
def create_matrix(rows,cols,fill=0):
    col_t = []
    for i in range(rows):
        row_t = []
        for j in range(cols):
            row_t.append(fill)
        col_t.append(row_t)
    return col_t

'''X x Y-transposed'''
def matrix_multipy(X,Y_t):
    '''Still need some more check, row-column restriction.
    I don't check because that is writing a new mathlib.
    I can just use numpy and I already planed to use it.
    '''

    is_X_vec = type(X[0]) == list
    is_Y_vec = type(Y_t[0]) == list

    # since this is X x Y_t, we only need to check equal length
    # zip with strict=true is better.

    if(not is_X_vec and not is_Y_vec):
        #print("vector x vector")
        return [
            sum(a*b for a,b in zip(X,Y_t,strict=True))
        ]

    elif(not is_X_vec and is_Y_vec):
        #print("vector x matrix")
        return [
            sum(a*b for a,b in zip(X,Y_col,strict=True))
            for Y_col in zip(*Y_t)
        ]    
    
    elif(is_X_vec and not is_Y_vec):
        #print("matrix x vector")
        return [
            sum(a*b for a,b in zip(X_row,Y_t,strict=True))
            for X_row in zip(X)
        ]    
    else:
        # matrix multiplied matrix
        return [
            [
                sum(a*b for a,b in zip(X_row,Y_col,strict=True)) 
                for Y_col in zip(*Y_t)
            ] 
                for X_row in X
        ]

class NeuralNetwork:
    def __init__(self, vec_in = [], vec_out = [], depth = 0):

        self.input = vec_in
        self.output = vec_out
        self.weight_t = []
        self.depth = depth

        if depth < 0:
            print("depth must not be negative")
        else:
            '''# Create Weight matrix filled with 1/sum(w).
            # because this is easier to rebalance.
            # The sum(sw) won't be obtained, but get sum(xw)/sum(w) right away.
            
            # vec_in, vec_out := row vector
            '''

            # initialization has simple weight of 1/len(vec_in)
            m = 1/len(vec_in)
            height = len(vec_in)
            width = height
            # number of matrixes match depth
            for l in range(depth):
                mat = create_matrix(height,width)
                for i in range(height):
                    for j in range(width):
                        mat[i][j] = m*(j%width+1) # test value
                        # mat[i][j] = m # real value
                self.weight_t.append(mat)
            
            # final matrix matches output
            out_height = len(vec_out)
            mat_fin = create_matrix(height,out_height)
            
            # test matrix
            for i in range(height):
                for j in range(out_height):
                    if(i==j):
                        mat_fin[i][j]=1

            self.weight_t.append(mat_fin)

    def run(self):
        
        # load the checkpoint.
        # -

        '''Internal States
        # Doesn't need to save into vector, just in memory calculation.
        The model obtain final state (output) by recursive calculation.
        This means no intermediate state requires.
        # The downside is cost of calculation.
        # Calculation is done by chained multiplication of weight matrix.
        '''
        # imediate state
        vec_im = self.input
        for i in range(self.depth+1):
            vec_im = matrix_multipy(vec_im, self.weight_t[i])
            print("L="+str(i))
            print(" ",vec_im)
            # self.print_matrix(vec_im)
        self.output = vec_im
        #print("result:", vec_im)

    def learn(self):
        
        # load the checkpoint.
        # -

        # calculate difference
        # -

        # adjust the model
        # rebalance weight matrix

        raise NotImplementedError("Not ready to learn, yet")
    
    def load_checkpoint(self):
        # File access
        # -

        # Set matrix to model
        # -

        raise NotImplementedError("Not ready to load checkpoint, yet.")
    
    def print_weight_t(self,padding = 6,precision = 2):
        if len(self.weight_t) == 0:
            print (self.weight_t)
            pass

        for i in range(self.depth+1):
            print("\nL"+ str(i) if (i != self.depth) else "fin")
            self.print_matrix(self.weight_t[i],padding,precision)

    # Print matrix prettier
    def print_matrix(self,mat,padding = 6,precision = 2):
        for row in mat:
            line = ""
            for val in row:
                line += format(float(val),'.{}f'.format(precision)).rjust(padding)+"\t"
            line = line[:-1]
            print(line)

x = NeuralNetwork(list(range(1,5)),list(range(1,4)),3)
print("input", x.input)
print("output", x.output)
# for i in range(x.depth+1):
#     print("weight {}:".format(i), x.weight_t[i])
print("depth = {}".format(x.depth))
x.print_weight_t()
print("run")
x.run()
print("output:", x.output)
#x.learn()
