from run import main

if __name__ == '__main__':
    for sigma in [0.1, 1, 2]:
        for j in range(5):
            print(f'(sigma={sigma}) ITERATION {j+1}----------------------------')
            main(True, sigma=sigma)
            print()
        print()