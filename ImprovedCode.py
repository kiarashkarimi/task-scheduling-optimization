import random
import math
import numpy as np

# Constants
MAX = 1000
NRUN = 20


# Task and Node classes
class Task:
    def __init__(self, id, s=0, m=0, d=0, p=0.0, q=0.0, in_size=0, out_size=0, resp=0):
        self.id = id
        self.s = s
        self.m = m
        self.d = d
        self.p = p
        self.q = q
        self.in_size = in_size
        self.out_size = out_size
        self.resp = resp


class Node:
    def __init__(self, id, c=0, m=0, b=0, a=0, d=0, pe=0.0, ce=0.0, pc=0.0, pmin=0.0, pmax=0.0):
        self.id = id
        self.c = c
        self.m = m
        self.b = b
        self.a = a
        self.d = d
        self.pe = pe
        self.ce = ce
        self.pc = pc
        self.pmin = pmin
        self.pmax = pmax
        self.eng = 0.0
        self.procCost = 0.0
        self.ms = 0.0
        self.fit = 0.0


# Task and Node initialization
def tasks(ntask):
    task_list = []
    for i in range(ntask):
        task = Task(i)
        x = random.randint(0, 2)
        if x == 0:
            task.s = random.randint(100, 10000)
            task.d = random.randint(100, 500)
        elif x == 1:
            task.s = random.randint(1028, 4280)
            task.d = random.randint(500, 2500)
        else:
            task.s = random.randint(5123, 9784)
            task.d = random.randint(2500, 10000)
        task.m = random.randint(50, 200)
        task.p = random.uniform(0.01, 0.5)
        task.q = random.uniform(9000.0, 10000.0) / 100.0
        task.in_size = random.randint(100, 10000)
        task.out_size = random.randint(1, 1000)
        task_list.append(task)
    return task_list


def nodes(nfog, ncloud):
    fog_nodes = []
    cloud_nodes = []

    for i in range(nfog):
        node = Node(i)
        node.c = random.randint(500, 1500)
        node.m = random.randint(150, 250)
        node.b = random.randint(10, 1000)
        node.pc = random.uniform(0.1, 0.4)
        node.d = random.randint(1, 10)
        node.pmax = random.uniform(40, 100)
        node.pmin = random.uniform(0.6, 1.0) * node.pmax
        fog_nodes.append(node)

    for i in range(ncloud):
        node = Node(i)
        node.c = random.randint(3000, 5000)
        node.m = random.randint(8192, 65536)
        node.b = random.randint(100, 10000)
        node.pc = random.uniform(0.7, 1.0)
        node.d = random.randint(200, 500)
        node.pmax = random.uniform(200, 400)
        node.pmin = random.uniform(0.6, 1.0) * node.pmax
        cloud_nodes.append(node)

    return fog_nodes, cloud_nodes


def sort_tasks(tasks):
    return sorted(tasks, key=lambda task: task.d)


# Scheduling Algorithms
def random_scheduling(task_list, fog, cloud, ntask, nfog, ncloud):
    sumM, sumengCons, sumprocCost, sumfitValue = 0, 0, 0, 0

    for r in range(NRUN):
        tasks = task_list.copy()
        fog, cloud = nodes(nfog, ncloud)
        min_M, min_engCons, min_procCost = calculate_min_values(tasks, fog, cloud)

        M = 0
        totalPenalty = 0
        PDST = 0
        violationCost = 0
        procCost = 0
        engCons = 0
        fitValue = 0

        for task in tasks:
            x = random.randint(0, nfog + ncloud - 1)
            if x < nfog:
                node = fog[x]
            else:
                node = cloud[x - nfog]

            etime = (float(task.s) / node.c) * 1000
            node.a += etime
            task.resp = node.a + node.d
            node.procCost += (float(task.s) / node.c * node.pc)

        for task in tasks:
            temp = task.resp - task.d
            if temp < 0:
                temp = 0
            tempq = temp * 100 / task.d - 100 + task.q
            if tempq > 0:
                violationCost += (tempq * task.p)

            if task.resp > task.d:
                totalPenalty += task.resp - task.d
            else:
                PDST += 1

        for node in fog + cloud:
            if node.a > M:
                M = node.a
            node.eng = (node.a / 1000.0) * node.pmax + ((M / 1000.0) - (node.a / 1000.0)) * node.pmin
            engCons += node.eng
            procCost += node.procCost

        # Avoid division by zero
        if engCons == 0:
            engCons = 1e-6  # Small value to avoid division by zero
        if procCost == 0:
            procCost = 1e-6  # Small value to avoid division by zero
        if M == 0:
            M = 1e-6  # Small value to avoid division by zero

        fitValue = 0.34 * min_engCons / engCons + 0.33 * min_procCost / procCost + 0.33 * min_M / M * 1000

        sumM += M
        sumengCons += engCons
        sumprocCost += procCost
        sumfitValue += fitValue

    print("\n*************Final Results*************")
    print(f"Makespan: {sumM / 1000.0 / NRUN}")
    print(f"engCons: {sumengCons / NRUN}")
    print(f"procCost: {sumprocCost / NRUN}")
    print(f"fitValue: {sumfitValue / NRUN}")


def calculate_min_values(tasks, fog, cloud):
    total_task_size = sum(task.s for task in tasks)
    total_cpu_fog = sum(node.c for node in fog)
    total_cpu_cloud = sum(node.c for node in cloud)

    # Calculate minimum makespan
    min_M = total_task_size / (total_cpu_fog + total_cpu_cloud)

    # Find the most power-efficient and cost-efficient nodes
    index_f_max_pe = max(fog, key=lambda node: node.c / node.pmax).id
    index_c_max_pe = max(cloud, key=lambda node: node.c / node.pmax).id
    index_f_max_ce = max(fog, key=lambda node: node.c / node.pc).id
    index_c_max_ce = max(cloud, key=lambda node: node.c / node.pc).id

    min_engCons = min(
        total_task_size / fog[index_f_max_pe].c * fog[index_f_max_pe].pmax,
        total_task_size / cloud[index_c_max_pe].c * cloud[index_c_max_pe].pmax,
    )

    min_procCost = min(
        total_task_size / fog[index_f_max_ce].c * fog[index_f_max_ce].pc,
        total_task_size / cloud[index_c_max_ce].c * cloud[index_c_max_ce].pc,
    )

    return min_M, min_engCons, min_procCost


# P2C Scheduling
def p2c_scheduling(task_list, fog, cloud, ntask, nfog, ncloud):
    sumM, sumengCons, sumprocCost, sumfitValue = 0, 0, 0, 0

    for r in range(NRUN):
        tasks = task_list.copy()
        fog, cloud = nodes(nfog, ncloud)
        min_M, min_engCons, min_procCost = calculate_min_values(tasks, fog, cloud)

        M = 0
        totalPenalty = 0
        PDST = 0
        violationCost = 0
        procCost = 0
        engCons = 0
        fitValue = 0

        for task in tasks:
            x = random.randint(0, nfog + ncloud - 1)
            y = random.randint(0, nfog + ncloud - 1)

            if x < nfog:
                fx = fog[x]
            else:
                fx = cloud[x - nfog]

            if y < nfog:
                fy = fog[y]
            else:
                fy = cloud[y - nfog]

            if fx.a + (task.s / fx.c) * 1000 < fy.a + (task.s / fy.c) * 1000:
                node = fx
            else:
                node = fy

            etime = (float(task.s) / node.c) * 1000
            node.a += etime
            task.resp = node.a + node.d
            node.procCost += (float(task.s) / node.c * node.pc)

        for task in tasks:
            temp = task.resp - task.d
            if temp < 0:
                temp = 0
            tempq = temp * 100 / task.d - 100 + task.q
            if tempq > 0:
                violationCost += (tempq * task.p)

            if task.resp > task.d:
                totalPenalty += task.resp - task.d
            else:
                PDST += 1

        for node in fog + cloud:
            if node.a > M:
                M = node.a
            node.eng = (node.a / 1000.0) * node.pmax + ((M / 1000.0) - (node.a / 1000.0)) * node.pmin
            engCons += node.eng
            procCost += node.procCost

        # Avoid division by zero
        if engCons == 0:
            engCons = 1e-6  # Small value to avoid division by zero
        if procCost == 0:
            procCost = 1e-6  # Small value to avoid division by zero
        if M == 0:
            M = 1e-6  # Small value to avoid division by zero

        fitValue = 0.34 * min_engCons / engCons + 0.33 * min_procCost / procCost + 0.33 * min_M / M * 1000

        sumM += M
        sumengCons += engCons
        sumprocCost += procCost
        sumfitValue += fitValue

    print("\n*************Final Results*************")
    print(f"Makespan: {sumM / 1000.0 / NRUN}")
    print(f"engCons: {sumengCons / NRUN}")
    print(f"procCost: {sumprocCost / NRUN}")
    print(f"fitValue: {sumfitValue / NRUN}")


# Pure Genetic Algorithm
def ga_scheduling(task_list, fog, cloud, ntask, nfog, ncloud):
    sumM, sumengCons, sumprocCost, sumfitValue = 0, 0, 0, 0
    mrate = 0.05  # Decreased mutation rate
    npop = 150  # Increased population size
    Nitr = 1000  # Increased iterations
    elitism_count = 2  # Keep the top 2 individuals

    for r in range(NRUN):
        tasks = task_list.copy()
        fog, cloud = nodes(nfog, ncloud)
        min_M, min_engCons, min_procCost = calculate_min_values(tasks, fog, cloud)

        # Initialize population
        sol = [[random.randint(0, nfog + ncloud - 1) for _ in range(ntask)] for _ in range(npop)]
        solM = [0] * npop
        solC = [0] * npop
        solOF = [0] * npop

        for itr in range(Nitr):
            # Evaluate fitness
            for i in range(npop):
                for node in fog + cloud:
                    node.a = 0
                    node.procCost = 0

                for j, task in enumerate(tasks):
                    x = sol[i][j]
                    if x < nfog:
                        node = fog[x]
                    else:
                        node = cloud[x - nfog]

                    etime = (float(task.s) / node.c) * 1000
                    node.a += etime
                    node.procCost += (float(task.s) / node.c * node.pc)

                solM[i] = max(node.a for node in fog + cloud)
                solC[i] = sum(node.procCost for node in fog + cloud)
                solOF[i] = 0.67 * min_M / solM[i] + 0.33 * min_procCost / solC[i]

            # Elitism: preserve top individuals
            sorted_indices = sorted(range(npop), key=lambda i: solOF[i], reverse=True)
            new_sol = [sol[i] for i in sorted_indices[:elitism_count]]

            # Selection
            sumF = sum(solOF)
            solOF = [sol / sumF for sol in solOF]
            for i in range(1, npop):
                solOF[i] += solOF[i - 1]

            # Crossover
            while len(new_sol) < npop:
                z1 = random.uniform(0, 1)
                z2 = random.uniform(0, 1)
                x = next(k for k in range(npop) if z1 <= solOF[k])
                y = next(k for k in range(npop) if z2 <= solOF[k])

                child1, child2 = sol[x][:ntask // 2] + sol[y][ntask // 2:], sol[y][:ntask // 2] + sol[x][ntask // 2:]
                new_sol.append(child1)
                if len(new_sol) < npop:
                    new_sol.append(child2)

            # Mutation
            for i in range(npop):
                if random.uniform(0, 1) < mrate:
                    j = random.randint(0, ntask - 1)
                    new_sol[i][j] = random.randint(0, nfog + ncloud - 1)

            sol = new_sol

        # Select the best solution
        best_sol_index = max(range(npop), key=lambda i: solOF[i])

        # Final assignment based on the best solution
        M, engCons, procCost = 0, 0, 0

        for node in fog + cloud:
            node.a = 0
            node.procCost = 0
            node.eng = 0

        for j, task in enumerate(tasks):
            x = sol[best_sol_index][j]
            if x < nfog:
                node = fog[x]
            else:
                node = cloud[x - nfog]

            etime = (float(task.s) / node.c) * 1000
            node.a += etime
            task.resp = node.a + node.d
            node.procCost += (float(task.s) / node.c * node.pc)

        for node in fog + cloud:
            node.eng = (node.a / 1000.0) * node.pmax + ((M / 1000.0) - (node.a / 1000.0)) * node.pmin
            engCons += node.eng
            procCost += node.procCost
            if node.a > M:
                M = node.a

        # Avoid division by zero
        if engCons == 0:
            engCons = 1e-6
        if procCost == 0:
            procCost = 1e-6
        if M == 0:
            M = 1e-6

        fitValue = 0.34 * min_engCons / engCons + 0.33 * min_procCost / procCost + 0.33 * min_M / M * 1000

        sumM += M
        sumengCons += engCons
        sumprocCost += procCost
        sumfitValue += fitValue

    print("\n*************Final Results*************")
    print(f"Makespan: {sumM / 1000.0 / NRUN}")
    print(f"engCons: {sumengCons / NRUN}")
    print(f"procCost: {sumprocCost / NRUN}")
    print(f"fitValue: {sumfitValue / NRUN}")


# Proposed ( High fit value, good respond time , hard to beat )
def proposed_scheduling(task_list, fog, cloud, ntask, nfog, ncloud):
    sumM, sumengCons, sumprocCost, sumfitValue = 0, 0, 0, 0

    for r in range(NRUN):
        tasks = task_list.copy()
        fog, cloud = nodes(nfog, ncloud)
        min_M, min_engCons, min_procCost = calculate_min_values(tasks, fog, cloud)

        M = 0
        totalPenalty = 0
        PDST = 0
        violationCost = 0
        procCost = 0
        engCons = 0
        fitValue = 0

        f_max_pe = max(fog, key=lambda x: x.pe).pe
        f_max_ce = max(fog, key=lambda x: x.ce).ce
        c_max_pe = max(cloud, key=lambda x: x.pe).pe
        c_max_ce = max(cloud, key=lambda x: x.ce).ce

        # Normalize power and cost efficiency for fog nodes
        if f_max_pe != 0:
            for node in fog:
                node.pe /= f_max_pe
        if f_max_ce != 0:
            for node in fog:
                node.ce /= f_max_ce

        # Normalize power and cost efficiency for cloud nodes
        if c_max_pe != 0:
            for node in cloud:
                node.pe /= c_max_pe
        if c_max_ce != 0:
            for node in cloud:
                node.ce /= c_max_ce

        for task in tasks:
            min_ms = float('inf')
            best_node = None

            for node in fog + cloud:
                etime = (float(task.s) / node.c) * 1000
                ms = node.a + etime

                if ms < min_ms:
                    min_ms = ms

            max_fit = -float('inf')

            for node in fog + cloud:
                etime = (float(task.s) / node.c) * 1000
                ms = (min_ms / (node.a + etime))
                fit = 0.25 * node.pe + 0.25 * node.ce + 0.5 * ms

                if fit > max_fit:
                    max_fit = fit
                    best_node = node

            best_node.a += (float(task.s) / best_node.c) * 1000
            task.resp = best_node.a + best_node.d
            best_node.procCost += (float(task.s) / best_node.c * best_node.pc)

        for task in tasks:
            temp = task.resp - task.d
            if temp < 0:
                temp = 0
            tempq = temp * 100 / task.d - 100 + task.q
            if tempq > 0:
                violationCost += (tempq * task.p)

            if task.resp > task.d:
                totalPenalty += task.resp - task.d
            else:
                PDST += 1

        for node in fog + cloud:
            node.eng = (node.a / 1000.0) * node.pmax + ((M / 1000.0) - (node.a / 1000.0)) * node.pmin
            engCons += node.eng
            procCost += node.procCost
            if node.a > M:
                M = node.a

        # Avoid division by zero
        if engCons == 0:
            engCons = 1e-6  # Small value to avoid division by zero
        if procCost == 0:
            procCost = 1e-6  # Small value to avoid division by zero
        if M == 0:
            M = 1e-6  # Small value to avoid division by zero

        fitValue = 0.34 * min_engCons / engCons + 0.33 * min_procCost / procCost + 0.33 * min_M / M * 1000

        sumM += M
        sumengCons += engCons
        sumprocCost += procCost
        sumfitValue += fitValue

    print("\n*************Final Results*************")
    print(f"Makespan: {sumM / 1000.0 / NRUN}")
    print(f"engCons: {sumengCons / NRUN}")
    print(f"procCost: {sumprocCost / NRUN}")
    print(f"fitValue: {sumfitValue / NRUN}")


# Simulated Annealing Algorithm

def simulated_annealing_scheduling(task_list, fog, cloud, ntask, nfog, ncloud):
    sumM, sumengCons, sumprocCost, sumfitValue = 0, 0, 0, 0
    initial_temp = 1000.0  # Starting temperature
    final_temp = 1.0  # Ending temperature
    alpha = 0.9  # Cooling rate
    num_iterations = 1000  # Number of iterations at each temperature level

    for r in range(NRUN):
        tasks = task_list.copy()
        fog, cloud = nodes(nfog, ncloud)
        min_M, min_engCons, min_procCost = calculate_min_values(tasks, fog, cloud)

        # Generate an initial solution randomly
        current_solution = [random.randint(0, nfog + ncloud - 1) for _ in range(ntask)]
        current_cost = evaluate_solution(tasks, fog, cloud, current_solution, min_M, min_engCons, min_procCost, nfog,
                                         ncloud)

        best_solution = current_solution[:]
        best_cost = current_cost

        current_temp = initial_temp

        while current_temp > final_temp:
            for _ in range(num_iterations):
                # Generate a neighbor solution by modifying the current solution
                neighbor_solution = current_solution[:]
                task_to_modify = random.randint(0, ntask - 1)
                neighbor_solution[task_to_modify] = random.randint(0, nfog + ncloud - 1)

                neighbor_cost = evaluate_solution(tasks, fog, cloud, neighbor_solution, min_M, min_engCons,
                                                  min_procCost, nfog, ncloud)

                # Calculate the acceptance probability
                diff = (current_cost - neighbor_cost) / current_temp
                diff = np.clip(diff, -700, 700)  # Clamp to prevent overflow in exp
                acceptance_prob = np.exp(diff)

                # Accept the neighbor solution with the calculated probability
                if neighbor_cost < current_cost or random.random() < acceptance_prob:
                    current_solution = neighbor_solution[:]
                    current_cost = neighbor_cost

                # Update the best solution found so far
                if current_cost < best_cost:
                    best_solution = current_solution[:]
                    best_cost = current_cost

            # Cool down the temperature
            current_temp *= alpha

        # Evaluate the final best solution
        final_M, final_engCons, final_procCost, final_fitValue = evaluate_solution(tasks, fog, cloud, best_solution,
                                                                                   min_M, min_engCons, min_procCost,
                                                                                   nfog, ncloud, return_all=True)

        sumM += final_M
        sumengCons += final_engCons
        sumprocCost += final_procCost
        sumfitValue += final_fitValue

    print("\n*************Final Results*************")
    print(f"Makespan: {sumM / 1000.0 / NRUN}")
    print(f"engCons: {sumengCons / NRUN}")
    print(f"procCost: {sumprocCost / NRUN}")
    print(f"fitValue: {sumfitValue / NRUN}")


# New, Proposed integrated with GA

def ga_proposed_scheduling(task_list, fog, cloud, ntask, nfog, ncloud):
    sumM, sumengCons, sumprocCost, sumfitValue = 0, 0, 0, 0
    mrate = 0.05  # Mutation rate
    npop = 150  # Population size
    Nitr = 1000  # Number of iterations
    elitism_count = 2  # Number of top individuals to keep

    for r in range(NRUN):
        tasks = task_list.copy()
        fog, cloud = nodes(nfog, ncloud)
        min_M, min_engCons, min_procCost = calculate_min_values(tasks, fog, cloud)

        # Initialize population
        sol = [[random.randint(0, nfog + ncloud - 1) for _ in range(ntask)] for _ in range(npop)]
        solM = [0] * npop
        solC = [0] * npop
        solOF = [0] * npop

        for itr in range(Nitr):
            # Evaluate fitness
            for i in range(npop):
                solM[i], solC[i], _, solOF[i] = evaluate_solution(tasks, fog, cloud, sol[i], min_M, min_engCons, min_procCost, nfog, ncloud, return_all=True)

            # Elitism: preserve top individuals
            sorted_indices = sorted(range(npop), key=lambda i: solOF[i], reverse=True)
            new_sol = [sol[i] for i in sorted_indices[:elitism_count]]

            # Selection
            sumF = sum(solOF)
            solOF = [sol / sumF for sol in solOF]
            for i in range(1, npop):
                solOF[i] += solOF[i - 1]

            # Crossover
            while len(new_sol) < npop:
                z1 = random.uniform(0, 1)
                z2 = random.uniform(0, 1)
                x = next(k for k in range(npop) if z1 <= solOF[k])
                y = next(k for k in range(npop) if z2 <= solOF[k])

                child1, child2 = sol[x][:ntask // 2] + sol[y][ntask // 2:], sol[y][:ntask // 2] + sol[x][ntask // 2:]
                new_sol.append(child1)
                if len(new_sol) < npop:
                    new_sol.append(child2)

            # Mutation
            for i in range(npop):
                if random.uniform(0, 1) < mrate:
                    j = random.randint(0, ntask - 1)
                    new_sol[i][j] = random.randint(0, nfog + ncloud - 1)

            # Apply the proposed scheduling logic to the best solutions
            for i in range(elitism_count):
                new_sol[i] = refine_solution_using_proposed(tasks, fog, cloud, new_sol[i], nfog, ncloud)

            sol = new_sol

        # Select the best solution
        best_sol_index = max(range(npop), key=lambda i: solOF[i])

        # Final assignment based on the best solution
        M, engCons, procCost = 0, 0, 0

        for node in fog + cloud:
            node.a = 0
            node.procCost = 0
            node.eng = 0

        for j, task in enumerate(tasks):
            x = sol[best_sol_index][j]
            if x < nfog:
                node = fog[x]
            else:
                node = cloud[x - nfog]

            etime = (float(task.s) / node.c) * 1000
            node.a += etime
            task.resp = node.a + node.d
            node.procCost += (float(task.s) / node.c * node.pc)

        for node in fog + cloud:
            node.eng = (node.a / 1000.0) * node.pmax + ((M / 1000.0) - (node.a / 1000.0)) * node.pmin
            engCons += node.eng
            procCost += node.procCost
            if node.a > M:
                M = node.a

        # Avoid division by zero
        if engCons == 0:
            engCons = 1e-6
        if procCost == 0:
            procCost = 1e-6
        if M == 0:
            M = 1e-6

        fitValue = 0.34 * min_engCons / engCons + 0.33 * min_procCost / procCost + 0.33 * min_M / M * 1000

        sumM += M
        sumengCons += engCons
        sumprocCost += procCost
        sumfitValue += fitValue

    print("\n*************Final Results*************")
    print(f"Makespan: {sumM / 1000.0 / NRUN}")
    print(f"engCons: {sumengCons / NRUN}")
    print(f"procCost: {sumprocCost / NRUN}")
    print(f"fitValue: {sumfitValue / NRUN}")


def refine_solution_using_proposed(tasks, fog, cloud, solution, nfog, ncloud):
    for task_idx, node_idx in enumerate(solution):
        task = tasks[task_idx]
        min_ms = float('inf')
        best_node = None

        for node in fog + cloud:
            etime = (float(task.s) / node.c) * 1000
            ms = node.a + etime

            if ms < min_ms:
                min_ms = ms

        max_fit = -float('inf')

        for node in fog + cloud:
            etime = (float(task.s) / node.c) * 1000
            ms = (min_ms / (node.a + etime))
            fit = 0.25 * node.pe + 0.25 * node.ce + 0.5 * ms

            if fit > max_fit:
                max_fit = fit
                best_node = node

        best_node.a += (float(task.s) / best_node.c) * 1000
        solution[task_idx] = best_node.id

    return solution


def evaluate_solution(tasks, fog, cloud, solution, min_M, min_engCons, min_procCost, nfog, ncloud, return_all=False):
    M = 0
    procCost = 0
    engCons = 0

    # Reset nodes
    for node in fog + cloud:
        node.a = 0
        node.procCost = 0
        node.eng = 0

    # Assign tasks to nodes according to the solution
    for task_idx, node_idx in enumerate(solution):
        if node_idx < nfog:
            node = fog[node_idx]
        else:
            node = cloud[node_idx - nfog]

        etime = (float(tasks[task_idx].s) / node.c) * 1000
        node.a += etime
        tasks[task_idx].resp = node.a + node.d
        node.procCost += (float(tasks[task_idx].s) / node.c * node.pc)

    # Calculate Makespan, Energy Consumption, and Processing Cost
    for node in fog + cloud:
        node.eng = (node.a / 1000.0) * node.pmax + ((M / 1000.0) - (node.a / 1000.0)) * node.pmin
        engCons += node.eng
        procCost += node.procCost
        if node.a > M:
            M = node.a

    # Avoid division by zero
    if engCons == 0:
        engCons = 1e-6  # Small value to avoid division by zero
    if procCost == 0:
        procCost = 1e-6  # Small value to avoid division by zero
    if M == 0:
        M = 1e-6  # Small value to avoid division by zero

    # Normalize each component
    normalized_M = min_M / M
    normalized_engCons = min_engCons / engCons
    normalized_procCost = min_procCost / procCost

    # Combine with adjusted weights
    fitValue = 0.34 * normalized_engCons + 0.33 * normalized_procCost + 0.33 * normalized_M

    if return_all:
        return M, engCons, procCost, fitValue
    else:
        return fitValue


# PSO Algorithm
def pso_scheduling(task_list, fog, cloud, ntask, nfog, ncloud):
    sumM, sumengCons, sumprocCost, sumfitValue = 0, 0, 0, 0
    num_particles = 50  # Number of particles
    max_iters = 100  # Maximum number of iterations
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient

    for r in range(NRUN):
        tasks = task_list.copy()
        fog, cloud = nodes(nfog, ncloud)
        min_M, min_engCons, min_procCost = calculate_min_values(tasks, fog, cloud)

        # Initialize particles
        particles = [np.random.randint(0, nfog + ncloud, ntask) for _ in range(num_particles)]
        velocities = [np.random.rand(ntask) for _ in range(num_particles)]
        personal_best_positions = particles.copy()
        personal_best_scores = [evaluate_solution(tasks, fog, cloud, p, min_M, min_engCons, min_procCost, nfog, ncloud)
                                for p in particles]
        global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
        global_best_score = max(personal_best_scores)

        for iter in range(max_iters):
            for i in range(num_particles):
                # Update velocity
                velocities[i] = (w * velocities[i] +
                                 c1 * np.random.rand(ntask) * (personal_best_positions[i] - particles[i]) +
                                 c2 * np.random.rand(ntask) * (global_best_position - particles[i]))

                # Update particle position
                particles[i] = np.clip(particles[i] + velocities[i], 0, nfog + ncloud - 1).astype(int)

                # Evaluate new position
                current_score = evaluate_solution(tasks, fog, cloud, particles[i], min_M, min_engCons, min_procCost,
                                                  nfog, ncloud)

                # Update personal best
                if current_score > personal_best_scores[i]:
                    personal_best_positions[i] = particles[i].copy()
                    personal_best_scores[i] = current_score

            # Update global best
            best_particle_index = np.argmax(personal_best_scores)
            if personal_best_scores[best_particle_index] > global_best_score:
                global_best_position = personal_best_positions[best_particle_index].copy()
                global_best_score = personal_best_scores[best_particle_index]

        # Evaluate the best solution found
        final_M, final_engCons, final_procCost, final_fitValue = evaluate_solution(tasks, fog, cloud,
                                                                                   global_best_position,
                                                                                   min_M, min_engCons, min_procCost,
                                                                                   nfog, ncloud, return_all=True)

        sumM += final_M
        sumengCons += final_engCons
        sumprocCost += final_procCost
        sumfitValue += final_fitValue

    print("\n*************Final Results*************")
    print(f"Makespan: {sumM / 1000.0 / NRUN}")
    print(f"engCons: {sumengCons / NRUN}")
    print(f"procCost: {sumprocCost / NRUN}")
    print(f"fitValue: {sumfitValue / NRUN}")


# The best Algorithm in this Case

# Enhanced Hybrid Genetic Algorithm (E-HGA)
def enhanced_hybrid_ga(task_list, fog, cloud, ntask, nfog, ncloud):
    sumM, sumengCons, sumprocCost, sumfitValue = 0, 0, 0, 0
    mrate = 0.05  # Mutation rate
    npop = 100  # Population size
    Nitr = 500  # Number of iterations
    elitism_count = 5  # Number of elite individuals to keep

    for r in range(NRUN):
        tasks = task_list.copy()
        fog, cloud = nodes(nfog, ncloud)
        min_M, min_engCons, min_procCost = calculate_min_values(tasks, fog, cloud)

        # Initialize population
        sol = [[random.randint(0, nfog + ncloud - 1) for _ in range(ntask)] for _ in range(npop)]
        solM = [0] * npop
        solC = [0] * npop
        solOF = [0] * npop

        for itr in range(Nitr):
            # Evaluate fitness
            for i in range(npop):
                solOF[i] = evaluate_solution(tasks, fog, cloud, sol[i], min_M, min_engCons, min_procCost, nfog, ncloud)

            # Elitism: preserve top individuals
            sorted_indices = sorted(range(npop), key=lambda i: solOF[i], reverse=True)
            new_sol = [sol[i] for i in sorted_indices[:elitism_count]]

            # Selection
            sumF = sum(solOF)
            solOF = [sol / sumF for sol in solOF]
            for i in range(1, npop):
                solOF[i] += solOF[i - 1]

            # Crossover
            while len(new_sol) < npop:
                z1 = random.uniform(0, 1)
                z2 = random.uniform(0, 1)
                x = next(k for k in range(npop) if z1 <= solOF[k])
                y = next(k for k in range(npop) if z2 <= solOF[k])

                child1, child2 = crossover(sol[x], sol[y], ntask, nfog, ncloud)
                new_sol.append(child1)
                if len(new_sol) < npop:
                    new_sol.append(child2)

            # Mutation
            for i in range(npop):
                if random.uniform(0, 1) < mrate:
                    mutate(new_sol[i], ntask, nfog, ncloud)

            sol = new_sol

        # Local search on the best solution
        best_sol_index = max(range(npop), key=lambda i: solOF[i])
        best_solution = sol[best_sol_index]
        best_solution = local_search(tasks, fog, cloud, best_solution, min_M, min_engCons, min_procCost, ntask, nfog, ncloud)

        # Evaluate the final best solution
        final_M, final_engCons, final_procCost, final_fitValue = evaluate_solution(tasks, fog, cloud, best_solution,
                                                                                   min_M, min_engCons, min_procCost,
                                                                                   nfog, ncloud, return_all=True)

        sumM += final_M
        sumengCons += final_engCons
        sumprocCost += final_procCost
        sumfitValue += final_fitValue

    print("\n*************Final Results*************")
    print(f"Makespan: {sumM / 1000.0 / NRUN}")
    print(f"engCons: {sumengCons / NRUN}")
    print(f"procCost: {sumprocCost / NRUN}")
    print(f"fitValue: {sumfitValue / NRUN}")


# Crossover function
def crossover(parent1, parent2, ntask, nfog, ncloud):
    cut = ntask // 2
    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]
    return child1, child2


# Mutation function
def mutate(solution, ntask, nfog, ncloud):
    task_to_modify = random.randint(0, ntask - 1)
    solution[task_to_modify] = random.randint(0, nfog + ncloud - 1)


# Local search function (similar to simulated annealing)
def local_search(tasks, fog, cloud, solution, min_M, min_engCons, min_procCost, ntask, nfog, ncloud):
    best_solution = solution[:]
    best_cost = evaluate_solution(tasks, fog, cloud, best_solution, min_M, min_engCons, min_procCost, nfog, ncloud)

    for _ in range(100):  # Local search iterations
        neighbor_solution = best_solution[:]
        task_to_modify = random.randint(0, ntask - 1)
        neighbor_solution[task_to_modify] = random.randint(0, nfog + ncloud - 1)
        neighbor_cost = evaluate_solution(tasks, fog, cloud, neighbor_solution, min_M, min_engCons, min_procCost, nfog, ncloud)

        if neighbor_cost > best_cost:  # Minimize the cost (maximize fitness)
            best_solution = neighbor_solution[:]
            best_cost = neighbor_cost

    return best_solution


# Evaluation function
def evaluate_solution(tasks, fog, cloud, solution, min_M, min_engCons, min_procCost, nfog, ncloud, return_all=False):
    M = 0
    procCost = 0
    engCons = 0

    # Reset nodes
    for node in fog + cloud:
        node.a = 0
        node.procCost = 0
        node.eng = 0

    # Assign tasks to nodes according to the solution
    for task_idx, node_idx in enumerate(solution):
        if node_idx < nfog:
            node = fog[node_idx]
        else:
            node = cloud[node_idx - nfog]

        etime = (float(tasks[task_idx].s) / node.c) * 1000
        node.a += etime
        tasks[task_idx].resp = node.a + node.d
        node.procCost += (float(tasks[task_idx].s) / node.c * node.pc)

    # Calculate Makespan, Energy Consumption, and Processing Cost
    for node in fog + cloud:
        node.eng = (node.a / 1000.0) * node.pmax + ((M / 1000.0) - (node.a / 1000.0)) * node.pmin
        engCons += node.eng
        procCost += node.procCost
        if node.a > M:
            M = node.a

    # Avoid division by zero
    if engCons == 0:
        engCons = 1e-6  # Small value to avoid division by zero
    if procCost == 0:
        procCost = 1e-6  # Small value to avoid division by zero
    if M == 0:
        M = 1e-6  # Small value to avoid division by zero

    fitValue = 0.34 * min_engCons / engCons + 0.33 * min_procCost / procCost + 0.33 * min_M / M * 1000

    if return_all:
        return M, engCons, procCost, fitValue
    else:
        return fitValue


# Main Program
def main():
    while True:
        ntask = int(input("\nEnter # of Tasks:\n"))
        nfog = 10
        ncloud = 5
        # Initialize tasks, fog nodes, and cloud nodes once per iteration
        tasks_list = tasks(ntask)
        fog_nodes, cloud_nodes = nodes(nfog, ncloud)

        while True:
            print("_________________________")
            print("\n1-Random")
            print("\n2-P2C")
            print("\n3-GA")
            print("\n4-Proposed")
            print("\n5-Start New Iteration")
            print("\n6-Simulated Annealing Algorithm - Kiarash")
            print("\n7-GA + Proposed - Kiarash")
            print("\n8-PSO Algorithm - Kiarash")
            print("\n9- E-HGA        - Kiarash")

            print("\n0 -Exit")
            code = int(input("\n\nEnter Your Choice: "))

            if code == 1:
                random_scheduling(tasks_list, fog_nodes, cloud_nodes, ntask, nfog, ncloud)
            elif code == 2:
                p2c_scheduling(tasks_list, fog_nodes, cloud_nodes, ntask, nfog, ncloud)
            elif code == 3:
                ga_scheduling(tasks_list, fog_nodes, cloud_nodes, ntask, nfog, ncloud)
            elif code == 4:
                proposed_scheduling(tasks_list, fog_nodes, cloud_nodes, ntask, nfog, ncloud)
            elif code == 5:
                # Start a new iteration: reinitialize tasks, fog nodes, and cloud nodes
                tasks_list = tasks(ntask)
                fog_nodes, cloud_nodes = nodes(nfog, ncloud)
            elif code == 6:
                simulated_annealing_scheduling(tasks_list, fog_nodes, cloud_nodes, ntask, nfog, ncloud)
            elif code == 7:
                ga_proposed_scheduling(tasks_list, fog_nodes, cloud_nodes, ntask, nfog, ncloud)
            elif code == 8:
                pso_scheduling(tasks_list, fog_nodes, cloud_nodes, ntask, nfog, ncloud)
            elif code == 9:
                enhanced_hybrid_ga(tasks_list, fog_nodes, cloud_nodes, ntask, nfog, ncloud)
            elif code == 0:
                exit(0)


if __name__ == "__main__":
    main()
