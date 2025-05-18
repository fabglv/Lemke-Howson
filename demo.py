import numpy as np

from lemke_howson import lemke_howson, show_final_results

def main():

    A = np.array([[3, 3],
                  [2, 5],
                  [0, 6]])

    B = np.array([[3, 2],
                  [2, 6],
                  [3, 1]])
    
    x, y = lemke_howson(A, B, 1, verbose = True)

    show_final_results(x,y)

if __name__ == "__main__":
    main()