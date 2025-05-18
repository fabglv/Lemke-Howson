import numpy as np

class Fraction:
    """A simple class to represent rational numbers for comparison in pivoting."""
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def __lt__(self,other):
        return self.numerator * other.denominator < other.numerator * self.denominator

def create_tables(A, B):
    """Creates the initial tables for both players' polytopes in the Lemke-Howson algorithm."""
    M, N = A.shape
    tables = [] 
    P_table = np.hstack([np.transpose(B), np.identity(N, dtype=int), np.ones((N,1), dtype=int)])
    Q_table = np.hstack([np.identity(M, dtype=int), A, np.ones((M,1), dtype=int)])
    P_table = P_table.astype(object)
    Q_table = Q_table.astype(object)
    tables = [P_table, Q_table]
    basic_vars = [set(range(M, M + N)), set(range(M))]
    return tables, basic_vars

def ratio_test(table, basic_vars, enter_var):
    """Performs the minimum ratio test to determine the pivot row and leaving variable."""
    min_ratio = Fraction(np.inf,1)
    for j in basic_vars:
        for i in range(table.shape[0]):
            if table[i,j] > 0 and table[i, enter_var] != 0:
                ratio = Fraction(table[i, -1],table[i, enter_var])
                if ratio < min_ratio:
                    min_ratio = ratio
                    row_min = i
                    col_min = j
                break
    return row_min, col_min #row_min is the pivot row, col_min is the leaving variable.

def pivoting(table, ip, jp, previous_pivot):
    """Performs a pivot operation on the tables."""
    # ip, jp are the coordinates of the current pivot.
    for i in range(table.shape[0]):
        if i != ip:
            table[i, :] = table[ip, jp] * table[i, :] - table[i, jp] * table[ip, :]
            table[i, :] = table[i, :] // previous_pivot # it's convenient to divide the rows here.
    return table

def lemke_howson(A, B, start_var, verbose=False):
    """Executes the Lemke-Howson algorithm to find a Nash equilibrium for a 2-player game."""
    M, N = A.shape
    tables, basic_vars = create_tables(A, B)

    if verbose:
        print('\n__________LEMKE_HOWSON__________\n')
        print('Starting Table for P:')
        print(tables[0])
        print('Basic variables: ', basic_vars[0])
        print('\nStarting Table for Q:')
        print(tables[1])
        print('Basic variables: ', basic_vars[1])
        print('\nStarting variable: ', start_var)

    player_dict = {0:'P', 1:'Q'}
    enter_var = start_var
    current = 0 # P
    if enter_var in basic_vars[0]:
        current = 1 # Q
    found = False
    iteration = 0
    previous_pivot = [1, 1]

    while not found and iteration <1000:

        for r in (0, 1):

            iteration += 1

            if verbose:
                print('________________________________\n')
                print('Iteration: ', iteration)

            k = (r + current) % 2
            
            if verbose:
                print('\nWorking with polytope ', player_dict[k])
                print('Entering variable: ', enter_var)
                print(tables[k])
                print('Basic variables: ', basic_vars[k])

            row_min, col_min = ratio_test(tables[k], basic_vars[k], enter_var)
            # pivot position = row_min, enter_var
            
            if verbose:
                print('\nLeaving variable: ', col_min)
                print('The pivot position is (', row_min, ',', enter_var, ')')

            tables[k] = pivoting(tables[k], row_min, enter_var, previous_pivot[k])
            previous_pivot[k] = tables[k][row_min, enter_var]
            basic_vars[k].add(enter_var)
            basic_vars[k].remove(col_min)
            enter_var = col_min
            
            if verbose:
                print('The pivot value is: ', previous_pivot[k])
                print('Table after pivoting and division:')
                print(tables[k])
                print('Basic variables: ', basic_vars[k])
            
            if enter_var == start_var:
                found = True
                break

    if verbose:
        print('\n__________FINAL_TABLES__________')
        print('\nPolytope ', player_dict[0])
        print(tables[0])
        print('Basic variables: ', basic_vars[0])
        print('\nPolytope ', player_dict[1])
        print(tables[1])
        print('Basic variables: ', basic_vars[1])
    
    strategy = [np.zeros(M+N), np.zeros(M+N)]
    for k in (0,1):
        col = tables[k][:, -1]
        for j in basic_vars[k]:
            for i in range(len(tables[k])):
                if tables[k][i][j] > 0:
                    strategy[k][j] = col[i]
        strategy[k]=strategy[k]/previous_pivot[k]

    x, y = strategy[0][:M], strategy[1][M:]
    x, y = x / np.sum(x), y / np.sum(y)

    return x, y

def show_final_results(x, y):
    """Prints the final strategies in a readable format."""
    print('\n__________RESULTS_______________')
    print(f'\nStrategy of P1 (x): {[round(e, 4) for e in x]}')
    print(f'Strategy of P2 (y): {[round(e, 4) for e in y]}\n')