import numpy as np
import matplotlib.pyplot as plt
import math  # Thêm import math

# Hàm mục tiêu ví dụ: hàm Sphere (tối ưu hóa về giá trị nhỏ nhất)
def objective_function(x):
    return np.sum(x**2)

# Levy flight
def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.randn() * sigma
    v = np.random.randn()
    step = u / abs(v) ** (1 / Lambda)
    return step

# Cuckoo Search Algorithm
def cuckoo_search(n=25, dim=2, lb=-10, ub=10, pa=0.25, max_iter=100):
    nests = np.random.uniform(lb, ub, size=(n, dim))
    fitness = np.array([objective_function(nest) for nest in nests])
    best_index = np.argmin(fitness)
    best = nests[best_index].copy()
    
    best_fitness_values = []  # Danh sách để lưu trữ giá trị fitness tốt nhất

    for iteration in range(max_iter):
        for i in range(n):
            new_nest = nests[i] + 0.01 * levy_flight(1.5) * (nests[i] - best)
            new_nest = np.clip(new_nest, lb, ub)
            f_new = objective_function(new_nest)

            if f_new < fitness[i]:
                nests[i] = new_nest
                fitness[i] = f_new

        # Loại bỏ một phần tổ trứng
        rand = np.random.rand(n, dim) > pa
        step_size = np.random.rand() * (nests[np.random.permutation(n)] - nests[np.random.permutation(n)])
        new_nests = nests + step_size * rand
        new_nests = np.clip(new_nests, lb, ub)

        for i in range(n):
            f_new = objective_function(new_nests[i])
            if f_new < fitness[i]:
                nests[i] = new_nests[i]
                fitness[i] = f_new

        # Cập nhật tổ tốt nhất
        best_index = np.argmin(fitness)
        best = nests[best_index]

        best_fitness_values.append(fitness[best_index])  # Lưu giá trị fitness tốt nhất
        print(f"Iter {iteration + 1}, Best fitness: {fitness[best_index]:.5f}")

    return best, fitness[best_index], best_fitness_values

# Chạy thuật toán
best_solution, best_value, best_fitness_values = cuckoo_search()

# Vẽ biểu đồ quá trình tối ưu hóa
plt.plot(best_fitness_values)
plt.title('Quá trình tối ưu hóa bằng thuật toán Cuckoo Search')
plt.xlabel('Số vòng lặp')
plt.ylabel('Giá trị fitness tốt nhất')
plt.grid()
plt.show()  # Đảm bảo rằng plt.show() được gọi

print("Best solution:", best_solution)
print("Best fitness value:", best_value)
