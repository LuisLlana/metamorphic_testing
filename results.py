from typing import Callable, List, Union, Dict
from qiskit import QuantumCircuit, QuantumRegister, execute, transpile
from qiskit.circuit.gate import Gate
from qiskit.providers.aer import Aer
from qiskit.circuit.quantumregister import Qubit
from adder_mod import grover,\
    fake_gate5, mtrule1 as mtR0, mtrule2 as mtR1, \
    get_sum_gate, get_sum_mutant2_gate
import numpy as np
import matplotlib.pyplot as plt
import json
import os


EXPERIMENTS_DIR = 'experiments'
R0_RES_FILE = f'{EXPERIMENTS_DIR}/sim_R0.txt'
R1_RES_FILE = f'{EXPERIMENTS_DIR}/sim_R1.txt'


def get_gates_R0(failures):
    nbits = len(failures[0])
    ngate_bits = 2*nbits+1
    gates = {}
    for i in range(len(failures)+1):
        gates[f'fakeR0_{i}'] = fake_gate5(nbits, failures[:i], ngate_bits = ngate_bits)
        if i == 0:
            gates[f'mtR0_{i}'] = mtR0(nbits, get_sum_gate(nbits))
        else:
            gates[f'mtR0_{i}'] = mtR0(nbits, get_sum_mutant2_gate(nbits, failures[:i]))
    return gates


def results_mtR0(failures) -> None:
    gates = get_gates_R0(failures)
    nbits = len(failures[0])
    ngate_bits = 2*nbits+1
    size = 2**nbits
    with open('R0_RES_FILE', 'w') as fout:
        sim = Aer.get_backend('aer_simulator')
        vector_sim = Aer.get_backend('statevector_simulator')
        for name, gate in gates.items():
            print(name)
            for steps in range(1, round(np.pi*np.sqrt(size)/4)):
                print(f'steps: {steps}')
                qc = grover(nbits, ngate_bits, gate.to_gate(), steps)
                counts = execute(qc, backend=sim, shots=size*100).result().get_counts()
                print(counts)
                line = [name, steps, counts]
                json.dump(line, fout)
                fout.write('\n')
                data  = execute(qc, backend=vector_sim).result().data()
                state_vector = data['psi']
                with open(f'{EXPERIMENTS_DIR}/{name}_{steps}.txt', 'w') as fvector:
                    for i in state_vector:
                        fvector.write(f'{i}\n')


def get_gates_R1(failures):
    nbits = len(failures[0])
    gates = {}
    for i in range(len(failures)+1):
        if i == 0:
            gates[f'mtR1_{i}'] = mtR1(nbits, get_sum_gate(nbits))
        else:
            gates[f'mtR1_{i}'] = mtR1(nbits, get_sum_mutant2_gate(nbits, failures[:i]))
    return gates


def results(failures:List[str],
            nbits: int,
            ngate_bits: int,
            get_gates: Callable[[List[str]], Dict[str, Gate]],
            outfilename: str
            ) -> None:

    gates = get_gates(failures)
    size = 2**nbits
    print(size)
    with open(outfilename, 'w') as fout:
        sim = Aer.get_backend('aer_simulator')
        vector_sim = Aer.get_backend('statevector_simulator')
        for name, gate in gates.items():
            print(name)
            for steps in range(1, int(np.ceil(np.pi*np.sqrt(size)/4))+1):
                print(f'steps: {steps}')
                qc = grover(nbits, ngate_bits, gate.to_gate(), steps,
                            save_statevector=False)
                qobj = transpile(qc, backend=sim)
                counts = execute(qobj, backend=sim, shots=size*100).result().get_counts()
                print(counts)
                line = [name, steps, counts]
                json.dump(line, fout)
                fout.write('\n')
                # data  = execute(qc, backend=vector_sim).result().data()
                # state_vector = data['psi']
                # with open(f'{EXPERIMENTS_DIR}/{name}_{steps}.txt', 'w') as fvector:
                #     for i in state_vector:
                #         fvector.write(f'{i}\n')


def get_values(data, nbits):
    values = [0]*2**nbits
    for key, v in data.items():
        pos = int(key, 2)
        values[pos] = v
    return values


def read_results(filename, nbits):
    with open(filename) as fin:
        results = {}
        for line in fin:
            data = json.loads(line)
            results[f'{data[0]}_{data[1]}'] = get_values(data[2], nbits)
    return results

def get_plot(values):
    values = np.array(values)
    ys = values/(len(values)*100)
    fig, ax = plt.subplots(figsize=(6*(len(values)/32), 4))
    mean = np.mean(ys)
    ind = range(len(ys))
    ax.bar(ind, ys)
    # ax.text(0,mean*1.1, mean)
    ax.hlines(mean, 0, len(ys), linestyles='dotted', color='red')
    ind = list(filter(lambda x: x%2**2==0, ind))
    ax.set_xticks(ind)
    labels = map(lambda x: f'{x:05b}', ind)
    ax.set_xticklabels(labels)
    return fig

def gen_charts(filename, nbits):
    results = read_results(filename, nbits)
    for k, v in results.items():
        fig = get_plot(v)
        fig.savefig(f'{EXPERIMENTS_DIR}/{k}.png')
        plt.close(fig)

def main():
    failures = ['10101', '10011', '10010', '11010']
    nbits = len(failures[0])
    results(failures, nbits, 2*nbits+1, get_gates_R0, R0_RES_FILE)
    gen_charts(R0_RES_FILE, nbits)

    failures = ['101', '100', '001']
    nbits = len(failures[0])
    print(R1_RES_FILE)
    results(failures, 2*nbits, 5*nbits+1, get_gates_R1, R1_RES_FILE)
    gen_charts(R1_RES_FILE, 2*nbits)


if __name__ == '__main__':
    if not os.path.isdir(EXPERIMENTS_DIR):
        os.mkdir(EXPERIMENTS_DIR)
    main()
