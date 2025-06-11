import numpy as np
import matplotlib.pyplot as plt

# Hàm mục tiêu cần tối ưu hóa (minimize)
def objective_function(x):
    return np.sum(x ** 2)  # Hàm Sphere

# Khởi tạo quần thể ruồi mayfly
def initialize_population(pop_size, dim, lb, ub):
    return lb + (ub - lb) * np.random.rand(pop_size, dim)

# Cập nhật vận tốc của ruồi
def update_velocity(v, x, pbest, gbest, a1, a2, beta, alpha):
    r1 = np.random.rand(*x.shape)
    r2 = np.random.rand(*x.shape)
    social = a1 * r1 * (pbest - x)
    global_ = a2 * r2 * (gbest - x)
    inertia = beta * v
    return inertia + social + global_ + alpha * (np.random.rand(*x.shape) - 0.5)

# Ràng buộc vị trí trong khoảng [lb, ub]
def bound_position(x, lb, ub):
    return np.clip(x, lb, ub)

# Main MOA
def mayfly_optimization(objective_function, dim=30, pop_size=50, max_iter=100,
                        lb=-5, ub=5, a1=1, a2=1, beta=0.7, alpha=0.1):
    male_size = pop_size // 2
    female_size = pop_size - male_size

    male_pos = initialize_population(male_size, dim, lb, ub)
    female_pos = initialize_population(female_size, dim, lb, ub)
    male_vel = np.zeros((male_size, dim))
    female_vel = np.zeros((female_size, dim))

    male_fit = np.array([objective_function(x) for x in male_pos])
    best_male_idx = np.argmin(male_fit)
    gbest = male_pos[best_male_idx].copy()
    
    best_fitness_history = []

    for iteration in range(max_iter):
        for i in range(male_size):
            male_vel[i] = update_velocity(male_vel[i], male_pos[i], male_pos[i], gbest, a1, a2, beta, alpha)
            male_pos[i] += male_vel[i]
            male_pos[i] = bound_position(male_pos[i], lb, ub)
            fit = objective_function(male_pos[i])
            if fit < male_fit[i]:
                male_fit[i] = fit
                if fit < objective_function(gbest):
                    gbest = male_pos[i].copy()

        for i in range(female_size):
            partner_idx = i % male_size
            direction = male_pos[partner_idx] - female_pos[i]
            female_vel[i] = beta * female_vel[i] + a1 * np.random.rand() * direction
            female_pos[i] += female_vel[i]
            female_pos[i] = bound_position(female_pos[i], lb, ub)

        best_fitness_history.append(objective_function(gbest))

        if iteration % 10 == 0 or iteration == max_iter - 1:
            print(f"Iteration {iteration}, Best Fitness: {objective_function(gbest):.5f}")

    return gbest, objective_function(gbest), best_fitness_history

# Chạy thử
best_solution, best_fitness, fitness_history = mayfly_optimization(objective_function)

# Hiển thị kết quả
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)

# Vẽ biểu đồ thể hiện quá trình tối ưu hóa
plt.figure(figsize=(12, 6))

# Biểu đồ phân tán (scatter plot)
plt.subplot(1, 2, 1)
plt.scatter(range(len(fitness_history)), fitness_history, color='blue', marker='o')
plt.title('Quá Trình Tối Ưu Hóa')
plt.xlabel('Iterations')
plt.ylabel('Giá Trị Fitness Tốt Nhất')
plt.grid()

# Biểu đồ cột (bar chart)
plt.subplot(1, 2, 2)
plt.bar(range(len(fitness_history)), fitness_history, color='orange')
plt.title('Biểu Đồ Cột Quá Trình Tối Ưu Hóa')
plt.xlabel('Iterations')
plt.ylabel('Giá Trị Fitness Tốt Nhất')
plt.grid()

plt.tight_layout()
plt.show()
