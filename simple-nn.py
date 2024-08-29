''' Simple Neural Network 
# Todo
[.] learning algorithm: rebalance network.
    [o] Weight rebalance
[o] evaluation: make output.
    [o] Matrix multiplication
[] data representation: taking inputs.
    [] Serialize data. (not necessary)
[] network description: saving result.
[] varies hidden layer: customizer.
'''

def create_matrix(rows,cols,fill=0):
    '''create filled matrix'''
    col_t = []
    for i in range(rows):
        row_t = []
        for j in range(cols):
            row_t.append(fill)
        col_t.append(row_t)
    return col_t

def matrix_transpose(X):
    if(type(X[0])!=list):
        return X
    
    rows = len(X)
    cols = len(X[0])
    temp = create_matrix(cols,rows)
    for i in range(rows):
        for j in range(cols):
            temp[j][i]=X[i][j]
    return temp


def matrix_multipy(X,Y):
    return [
        [
            sum(a*b for a,b in zip(X_row,Y_col,strict=True)) 
            for Y_col in zip(*Y)
        ] 
            for X_row in X
    ]

# Print matrix prettier
def print_matrix(mat,padding = 6,precision = 2):
    for row in mat:
        line = ""
        for val in row:
            line += format(float(val),'.{}f'.format(precision)).rjust(padding)+"\t"
        line = line[:-1]
        print(line)


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
                # W_t row
                for i in range(height):
                    # W_t col
                    for j in range(width):
                        # mat[i][j] = m*(j%width+1) # test value
                        mat[i][j] = (i*width+j+1)/(width*height)
                        # mat[i][j] = m # real value
                self.weight_t.append(mat)
            
            # final matrix matches output
            out_height = len(vec_out)
            mat_fin = create_matrix(height,out_height)
            
            # test matrix
            for i in range(height):
                for j in range(out_height):
                    if(i==j):
                        mat_fin[i][j] = 1
                        mat_fin[(height-i-1) % height][j] = m
                        # diagonal down = 1, diagonal up = 0.5
                        # why not [j][i]? because memory out-of-bound

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
        vec_im = [self.input]
        print("vec_im",vec_im)
        for i in range(self.depth+1):
            vec_im = matrix_multipy(vec_im, self.weight_t[i])
            print("L="+str(i))
            print(" ",vec_im)
            # print_matrix(vec_im)
        self.output = vec_im[0]
        #print("result:", vec_im)

    def learn(self):
        
        # load the checkpoint.
        # -

        # load datasets
        ''' Serialize inputs
        loop through files, and entries
        maybe jsonl.
        What kind of data? What context?
        The network should be able to auto scale vertically.
        Because neuron is float64, this means any input should
        be in in 8 bytes. (Python doesn't have this limit, but
        but I just want it to be true.)
        The real goal of this AI is to have AI work with real
        world. So Webcam, Microphone. But, it's impossible.
        My PC can't move. The only working solution is to put
        it on phone. Compile for Android, or have my AI server.
        '''

        # calculate difference
        # -

        # adjust the model
        self.rebalance()

        #raise NotImplementedError("Not ready to learn, yet")
    
    def rebalance(self):
        # rebalance weight matrix
        for layer in range(self.depth+1):
            Mat = self.weight_t[layer]

            # summing node weights
            summer = None
            width = len(Mat)
            summer = list(range(width))

            print("L{}".format(layer))

            for item in summer: 
                summer[item] = 1

            print("summer", summer)

            w_sum = matrix_multipy([summer], Mat)
            print("Sum W", w_sum)

            # divide weight matrix with collumn sum
            Mat_t = Mat
            t_width = len(Mat_t)
            t_height = len(Mat_t[0])
            for i in range(t_width):
                for j in range(t_height):
                    Mat_t[i][j] /= w_sum[0][j]
            print("renorm")
            print_matrix(Mat_t)

    def load_checkpoint(self):
        # File access
        # -

        # Set matrix to model
        # -

        raise NotImplementedError("Not ready to load checkpoint, yet.")
    
    def print_weight_t(self,padding = 6,precision = 2):
        if len(self.weight_t) == 0:
            print(self.weight_t)
            pass

        for i in range(self.depth+1):
            print("\nL"+ str(i) if (i != self.depth) else "\nfin")
            print_matrix(self.weight_t[i],padding,precision)


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
print("\nlearn")
x.learn()
# for i in range(x.depth+1):
#     print("weight {}:\n".format(i), x.weight_t[i])