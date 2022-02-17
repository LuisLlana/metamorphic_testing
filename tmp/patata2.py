from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import Qubit
from qiskit import transpile, assemble, execute
from qiskit.providers.aer import Aer
from qiskit.circuit.library import MCXGate
from qiskit.tools.visualization import plot_histogram
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import Union
from typing import Callable
from typing import List
from typing import Dict
from io import TextIOBase

AER_SIM = Aer.get_backend('aer_simulator')

def patata():
    print("patata")

def patata2():
    print("zanahoria")
    
def add_sumbits(qc: QuantumCircuit,
                s1: Union[int, Qubit],
                s2: Union[int, Qubit],
                res: Union[int, Qubit],
                anc: Union[int, Qubit]) -> None:
    if anc is not None:
        qc.ccx(s1, res, anc)
    qc.cx(s1, res)
    if anc is not None:
        qc.ccx(s2, res, anc)
    qc.cx(s2, res)


def test_1():
    qc = QuantumCircuit(4, 2)
    qc.x(1)
    qc.x(3)
    qc.barrier()
    add_sumbits(qc, 0, 1, 2, 3)
    qc.barrier()
    qc.measure([2, 3], [0, 1])
    qc.draw(output='mpl')


def test_2():
    qc = QuantumCircuit(4, 2)
    qc.x(1)
    qc.x(3)
    qc.barrier()
    add_sumbits(qc, 0, 1, 2, 3)
    qc.barrier()
    qc.measure([2, 3], [0, 1])
    sim = Aer.get_backend('aer_simulator')
    qobj = assemble(transpile(qc, backend=sim))
    job = sim.run(qobj)
    res = job.result()
    counts = res.get_counts()
    print(counts)


def test_sumbits():
    tests = [(s1, s2)
             for s1 in range(2)
             for s2 in range(2)]
    sim = Aer.get_backend('aer_simulator')
    for s1, s2 in tests:
        qc = QuantumCircuit(4, 2)
        if s1:
            qc.x(0)
        if s2:
            qc.x(1)
        add_sumbits(qc, 0, 1, 2, 3)
        qc.measure([2, 3], [0, 1])
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj)
        res = job.result()
        counts = list(res.get_counts())
        print(s1, s2, counts, end='...')
        assert s1+s2 == int(counts[0], 2), 'error'
        print('ok')


def add_int(qc, n, pos):
    while n != 0:
        if n % 2 == 1:
            qc.x(pos)
        pos += 1
        n = n//2


def view_sumbits_1():
    nbits = 3
    qs1 = QuantumRegister(nbits, 's1')
    qs2 = QuantumRegister(nbits, 's2')
    qres = QuantumRegister(nbits, 'res')
    qanc = QuantumRegister(1, 'anc')
    cres = ClassicalRegister(nbits, 'cres')
    canc = ClassicalRegister(1, 'canc')
    qc = QuantumCircuit(qs1, qs2, qres, qanc, cres, canc)
    add_int(qc, 3, 0)
    add_int(qc, 3, nbits)
    qc.barrier()
    add_sumbits(qc, 0, 2, 4, 5)
    qc.barrier()
    add_sumbits(qc, 1, 3, 5, 6)
    qc.barrier()
    qc.measure(qres, cres)
    qc.measure(qanc, canc)
    return qc


def sim_sumbits():
    qc = view_sumbits_1()
    counts = Aer.get_backend('aer_simulator').run(qc).result().get_counts()
    print(counts)


def view_sumbits_2():
    nbits = 2
    qs1 = QuantumRegister(nbits, 's1')
    qs2 = QuantumRegister(nbits, 's2')
    qres = QuantumRegister(nbits, 'res')
    qanc = QuantumRegister(1, 'anc')
    cres = ClassicalRegister(nbits, 'cres')
    canc = ClassicalRegister(1, 'canc')
    qc = QuantumCircuit(qs1, qs2, qres, qanc, cres, canc)
    add_int(qc, 3, 0)
    add_int(qc, 3, nbits)
    qc.barrier()
    add_sumbits(qc, 0, 2, 4, 5)
    qc.barrier()
    add_sumbits(qc, 1, 3, 5, None)
    qc.barrier()
    qc.measure(qres, cres)
    qc.measure(qanc, canc)
    qc.draw(output='mpl')
    plt.show()


def sumnbits_circuit(nbits: int,
                     barrier: bool = False) -> QuantumCircuit:
    qs1 = QuantumRegister(nbits, 's1')
    qs2 = QuantumRegister(nbits, 's2')
    qres = QuantumRegister(nbits, 'res')
    # qanc = QuantumRegister(1, 'anc')
    # cres = ClassicalRegister(nbits, 'cres')
    # canc = ClassicalRegister(1, 'canc')
    qc = QuantumCircuit(qs1, qs2, qres)

    s1 = qs1[:]
    s2 = qs2[:]
    res = qres[:]

    if barrier:
        qc.barrier()
    for j in range(nbits):
        if j == nbits-1:
            anc = None
        else:
            anc = res[j+1]
        add_sumbits(qc, s1[j], s2[j], res[j], anc)
        # add_sum_mutant_bits(qc, s1+j, s2+j, res+j, anc)
        if barrier:
            qc.barrier()
    return qc


def view_sumnbits_circuit():
    nbits = 4
    qc = sumnbits_circuit(nbits, barrier=True)
    # qc.draw(output='latex', filename='adder.tex')
    print(qc.draw(fold=-1))


def get_sum_gate(nbits: int) -> Gate :
    qsum = sumnbits_circuit(nbits)
    sgate = qsum.to_gate(label=' sum_nbits ')
    return sgate


def test_sumnbits():
    nbits = 3
    tests = [(n1, n2) for n1 in range(2**nbits) for n2 in range(2**nbits)]
    sim = Aer.get_backend('aer_simulator')
    sgate = get_sum_gate(nbits)
    for n1, n2 in tests:
        qc = QuantumCircuit(3*nbits, nbits)
        add_int(qc, n1, 0)
        add_int(qc, n2, nbits)
        qc.append(sgate, range(3*nbits))
        qc.measure(range(2*nbits, 3*nbits), range(nbits))
        qobj = assemble(transpile(qc, backend=sim))
        counts = sim.run(qobj, shots=1000).result().get_counts()
        print(n1, n2, counts, end='...')
        res = int(list(counts)[0], 2)
        assert res == (n1+n2) % (2**nbits), \
            f'error in n1={n1}, n2={n2}, res={res}'
        print('ok')


def test_sumnbits_2():
    nbits = 3
    tests = [(n1, n2, res) for n1 in range(2**nbits) for n2 in range(2**nbits)
             for res in range(2**nbits)]
    sgate = get_sum_gate(nbits)
    sim = Aer.get_backend('aer_simulator')
    for n1, n2, res in tests:
        qc = QuantumCircuit(3*nbits)
        add_int(qc, n1, 0)
        add_int(qc, n2, nbits)
        add_int(qc, res, 2*nbits)
        qc.append(sgate, range(3*nbits))
        qc.append(sgate.inverse(), range(3*nbits))
        qc.measure_all()
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj)
        counts = list(job.result().get_counts())
        resr = int(counts[0][:nbits], 2)
        n2r = int(counts[0][nbits:nbits*2], 2)
        n1r = int(counts[0][nbits*2:], 2)
        print(n1, n2, res, end='...')
        assert len(counts) == 1 and res == resr and n1r == n1 and n2r == n2, \
            f'error in n1={n1}, n2={n2}, res={res}'
        print('ok')


def succ_gate(nbits):
    qc = QuantumCircuit(2*nbits)
    res = nbits
    qc.x(res)
    for i in range(nbits-1):
        qc.ccx(i, res+i, res+i+1)
        qc.cx(i, res+i)
    qc.cx(nbits-1, 2*nbits-1)
    us = qc.to_gate(label=' succ ')
    return us



def test_succ(nbits):
    tests = [ n1 for n1 in range(2**nbits) ]
    sgate = succ_gate(nbits)
    for n1 in tests:
        qc  = QuantumCircuit(2*nbits, nbits)
        add_int(qc, n1, 0)
        qc.append(sgate, range(2*nbits))
        qc.measure(range(nbits, 2*nbits), range(nbits))
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj)
        counts = list(job.result().get_counts())
        print(n1, end='....')
        assert len(counts) == 1, 'error, more than one result'
        res = int(counts[0], 2)
        print(res, end='...')
        assert  res == (n1+1)%2**nbits, f'error n1={n1}, res={int(counts[0], 2)}'
        print('ok')


'''
  These mehtod is mine idea. It has been taken from..... (I have to look for the reference)
'''
def add_maj(qc: QuantumCircuit,
            c: Union[int, Qubit],
            b: Union[int, Qubit],
            a: Union[int, Qubit]) -> None:
    qc.cx(a, b)
    qc.cx(a, c)
    qc.ccx(c, b, a)


def add_uma(qc: QuantumCircuit,
            c: Union[int, Qubit],
            b: Union[int, Qubit],
            a: Union[int, Qubit]) -> None:
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.cx(c, b)

def view_maj_uma() -> None:
    qc = QuantumCircuit(3)
    c, b, a = 0, 1, 2
    add_maj(qc, c, b, a)
    add_uma(qc, c, b, a)
    return qc

def view_maj() -> None:
    a = QuantumRegister(1, 'a')
    b = QuantumRegister(1, 'b')
    c = QuantumRegister(1, 'c')
    qc = QuantumCircuit(c, b, a)
    add_maj(qc, c, b, a)
    return qc

def view_uma() -> None:
    a = QuantumRegister(1, 'a')
    b = QuantumRegister(1, 'b')
    c = QuantumRegister(1, 'c')
    qc = QuantumCircuit(c, b, a)
    add_uma(qc, c, b, a)
    return qc


def get_maj_gate() -> Gate:
   qc = QuantumCircuit(3)
   add_maj(qc, 0, 1, 2)
   gate = qc.to_gate(label=' MAJ ', )
   return gate

def get_uma_gate() -> Gate:
   qc = QuantumCircuit(3)
   add_uma(qc, 0, 1, 2)
   gate = qc.to_gate(label=' UMA ')
   return gate


def sum_inplace_circuit(nbits:int,
                        barrier: bool = False) -> QuantumCircuit:
    ar = QuantumRegister(nbits, 'a')
    br = QuantumRegister(nbits, 'b')
    ancr = QuantumRegister(1, 'anc')
    qc = QuantumCircuit(ancr, br, ar)

    maj = get_maj_gate()
    uma = get_uma_gate()
    a = ar[:]
    b = br[:]
    anc = ancr[:][0]

    for i in range(nbits):
        if i == 0:
            c = anc
        else:
            c = a[i-1]

        qc.append(maj, [c, b[i], a[i]])
        # add_maj(qc, c, b+i, a+i)
        if barrier:
            qc.barrier()
    for i in range(nbits-1, -1, -1):
        if i == 0:
            c = anc
        else:
            c = a[i-1]
        qc.append(uma, [c, b[i], a[i]])
        # add_uma(qc, c, b+i, a+i)
        if barrier:
            qc.barrier()
    return qc

def get_sum_inplace_gate(nbits):
    qc = sum_inplace_circuit(nbits, barrier=False)
    gate = qc.to_gate(label=' sum_i ')
    return gate


def test_sum_inplace():
    nbits = 4
    tests = [(n1, n2) for n1 in range(2**nbits) for n2 in range(2**nbits)]
    sgate = get_sum_inplace_gate(nbits)
    sim = Aer.get_backend('aer_simulator')
    for a, b in tests:
        qc = QuantumCircuit(2*nbits+1)
        add_int(qc, b, 1)
        add_int(qc, a, nbits+1)
        qc.append(sgate, range(2*nbits+1))
        qc.measure_all()
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj)
        counts = list(job.result().get_counts())
        print(a, b, end='...')
        assert len(counts) == 1, f'error len(counts)={len(counts)}'
        print(counts[0], end='...')
        ar = int(counts[0][:nbits], 2)
        br = int(counts[0][nbits:2*nbits], 2)
        cr = int(counts[0][2*nbits], 2)
        assert cr == 0, f'error cr={cr}'
        print(cr, end='...')
        assert a == ar, f'error n1r={ar}'
        assert (a+b) % 2**nbits == br, f'error br={b}'
        print(ar, br, end='...')
        print('ok')


def test_sum_inplace2(nbits):
    tests = [(n1, n2) for n1 in range(2**nbits) for n2 in range(2**nbits)]
    sgate = get_sum_inplace_gate(nbits)
    sim = Aer.get_backend('aer_simulator')
    for n1, n2 in tests:
        #qc  = init_sumcircuit(nbits)
        qc = QuantumCircuit(2*nbits+1)
        add_int(qc, n1, 1)
        add_int(qc, n2, nbits+1)
        qc.append(sgate, range(2*nbits+1))
        qc.append(sgate.inverse(), range(2*nbits+1))
        #qc.measure(range(2*nbits, 3*nbits), range(nbits))
        qc.measure_all()
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj)
        counts = list(job.result().get_counts())
        res = int(counts[0][nbits*2], 2)
        n1r = int(counts[0][nbits:nbits*2], 2)
        n2r = int(counts[0][:nbits], 2)
        print(n1, n2, end='...')
        assert len(counts) == 1 and res == 0 and n1r == n1 and n2r == n2, f'error in n1r={n1r}, n2r={n2r}, res={res}'
        print('ok')


# In[ ]:


# test_sum_inplace2(3)


# In[ ]:


def eq_circuit(nbits: int,
               barrier: bool = False) -> QuantumCircuit:
    qa = QuantumRegister(nbits, 'a')
    qb = QuantumRegister(nbits, 'b')
    qanc = QuantumRegister(1, 'anc')
    qres = QuantumRegister(1, 'res')
    qc = QuantumCircuit(qanc, qb, qa, qres)

    res = qres[:]
    a = qa[:]
    b = qb[:]

    sgate = get_sum_inplace_gate(nbits)
    qc.x(b)
    # for i in range(nbits):
    #     qc.x(b+i)
    qc.append(sgate, range(2*nbits+1))
    qc.append(MCXGate(nbits), b + res)
    suminv = sgate.inverse()
    qc.append(suminv, range(2*nbits+1))
    qc.x(b)
    # for i in range(nbits):
    #     qc.x(b+i)
    return qc

def eq_gate(nbits: int) -> Gate:
    qc = eq_circuit(nbits)
    gate = qc.to_gate(label=' eq ')
    return gate


def test_eq():
    n = 3
    tests = [(n1, n2) for n1 in range(2**nbits) for n2 in range(2**nbits) ]
    sim = Aer.get_backend('aer_simulator')
    eq = eq_gate(nbits)
    for n1, n2 in tests:
        qc = QuantumCircuit(2*nbits+2, 2*nbits+2)
        add_int(qc, n1, 1)
        add_int(qc, n2, nbits+1)
        #add_int(qc, res, 2*nbits)
        qc.append(eq, range(2*nbits+2))
        #add_eq(qc, nbits)
        qc.measure(range(2*nbits+2), range(2*nbits+2))
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj)
        counts = list(job.result().get_counts())
        print(n1, n2, end='...')
        assert len(counts) == 1, 'error counts={counts}'
        res = int(counts[0][0], 2)
        ancr = int(counts[0][2*nbits+1], 2)
        n2r =  int(counts[0][1:nbits+1], 2)
        n1r =  int(counts[0][nbits+1:2*nbits+1], 2)
        print(counts[0], ancr, n1r, n2r, res, end='....')
        assert ancr == 0, f'error,  ancr={ancr}'
        assert (n1 == n1r), f'error,  n1r={n1r}'
        assert n2 == n2r, f'error,  n2r={n2r}'
        assert (n1 == n2) == res, f'error,  res={res}'
        print('ok')


# The metamorpic rule $x+0=x$


def mtrule1(nbits: int,
            sumgate: Gate,
            barrier: bool = False) -> Gate:
    qxr = QuantumRegister(nbits, 'x')
    qzeror = QuantumRegister(nbits, 'zero')
    sumr = QuantumRegister(nbits, 'sum')
    resr = QuantumRegister(1, 'res')
    qanc = QuantumRegister(1, 'anc')
    qc = QuantumCircuit(qxr, resr, qzeror, sumr, qanc)
    qc.append(sumgate, qxr[:]+qzeror[:]+sumr[:])
    if barrier:
        qc.barrier()
    qc.append(eq_gate(nbits), qanc[:]+qxr[:]+sumr[:]+resr[:])
    if barrier:
        qc.barrier()
    qc.append(sumgate.inverse(), qxr[:]+qzeror[:]+sumr[:])
    qc.x(resr)  # we want to find the failed tests
    return qc



def get_mtrule1_gate(nbits:int, sumgate: Gate) -> Gate:
    qc = mtrule1(nbits, sumgate, barrier=False)
    return qc.to_gate(label='mtrule1')


def test_mtrule1_gate() -> None:
    nbits = 4
    sumgate = get_sum_gate(nbits)
    tests = range(2**nbits)
    sim = Aer.get_backend('aer_simulator')
    rule_gate = get_mtrule1_gate(nbits, sumgate)
    for t in tests:
        qc = QuantumCircuit(3*nbits+2, 1)
        add_int(qc, t, 0)
        qc.append(rule_gate, range(3*nbits+2))
        qc.measure([nbits], [0])
        job = execute(qc, backend=sim)
        counts = list(job.result().get_counts())
        print(f'{t:04b}, {int(counts[0], 2)}', end='...')
        res = int(counts[0], 2)
        assert res == 0, 'error'
        print('ok')

# Testing the metamorfic rule in the classical way

def exec_mtrule1_gate(nbits: int, sumgate: Gate) -> None:
    tests = range(2**nbits)
    sim = Aer.get_backend('aer_simulator')
    rule_gate = get_mtrule1_gate(nbits, sumgate)
    for t in tests:
        qc = QuantumCircuit(3*nbits+2, 1)
        add_int(qc, t, 0)
        qc.append(rule_gate, range(3*nbits+2))
        qc.measure([nbits], [0])
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj)
        counts = list(job.result().get_counts())
        print(f'{t:04b}, {int(counts[0], 2)}', end='...')
        res = int(counts[0], 2)
        if res:
            print('Fail')
        else:
            print('OK')


def extended_oracle(oracle: QuantumCircuit) -> QuantumCircuit:
    """
    This oracle extends the original one. It adds a new qubit so that the
    search space is duplicated. All the new elements are no solutions.
    """
    nbits = len(oracle.qubits)
    qc = QuantumCircuit(nbits + 1)
    qc.append(oracle.control(1), range(nbits + 1))
    return qc

def exec_mtrule1_gate_extended(nbits: int, sumgate: Gate) -> None:
    """
    The exedend mtrule1, the search space is duplicated.
    """
    tests = range(2**(nbits+1))
    sim = Aer.get_backend('aer_simulator')
    egate = extended_oracle(mtrule1(nbits, sumgate))
    for t in tests:
        qc = QuantumCircuit(3*nbits+3, 1)
        add_int(qc, t, 0)
        qc.append(egate, range(3*nbits+3))
        qc.measure([nbits+1], [0])
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj)
        counts = list(job.result().get_counts())
        print(f'{t:04b}, {int(counts[0], 2)}', end='...')
        res = int(counts[0], 2)
        if res:
            print('Fail')
        else:
            print('OK')


def add_sum_mutant1_bits(qc: QuantumCircuit,
                         s1: Union[int, Qubit],
                         s2: Union[int, Qubit],
                         res: Union[int, Qubit],
                         anc: Union[int, Qubit]) -> None:
    if anc is not None:
        qc.ccx(s1, res, anc)
    qc.cx(res, s1)  # mutant
    if anc is not None:
        qc.ccx(s2, res, anc)
    qc.cx(s2, res)

def add_sum_mutant2_bits(qc: QuantumCircuit,
                         s1: Union[int, Qubit],
                         s2: Union[int, Qubit],
                         res: Union[int, Qubit],
                         anc: Union[int, Qubit]) -> None:
    if anc is not None:
        qc.ccx(s1, res, anc)
    qc.cx(s1, res)
    # qc.cx(res, s1) #mutant: change gates
    # if anc is not None: #mutant: remove the conditional
    #     qc.ccx(s2, res, anc)
    qc.cx(s2, res)

# sumbits_mutants = [add_sum_mutant1_bits, add_sum_mutant2_bits]

def add_sum_mutant1_nbits(qc:QuantumCircuit,
                          nbits: int,
                          add_sum_mutant_bits: Callable[[QuantumCircuit,
                                                         Union[int, Qubit],
                                                         Union[int, Qubit],
                                                         Union[int, Qubit],
                                                         Union[int, Qubit]], None],
                          barrier: bool=True):
    s1 = 0
    s2 = nbits
    res = 2*nbits
    anc = 3*nbits
    if barrier:
        qc.barrier()
    for j in range(nbits):
        if j == nbits-1:
            anc = None
        else:
            anc = res+j+1
        if j != nbits//2:
            add_sumbits(qc, s1+j, s2+j, res+j, anc)
        else:
            add_sum_mutant_bits(qc, s1+j, s2+j, res+j, anc)
        if barrier:
            qc.barrier()


def sum_mutant2_nbits(nbits: int,
                      fail_values: List[str] = [],
                      barrier: bool = False) -> QuantumCircuit:
    qc = sumnbits_circuit(nbits, barrier=barrier)
    for value in fail_values:
        for i in range(nbits):
            if value[i] == '0':
                qc.x(nbits-1-i)
        qc.append(MCXGate(nbits), list(range(nbits))+[2*nbits])
        for i in range(nbits):
            if value[i] == '0':
                qc.x(nbits-1-i)
        if barrier:
            qc.barrier()
    return qc


def get_sum_mutant2_gate(nbits: int, fail_values: List[str]) -> Gate:
    qc = sum_mutant2_nbits(nbits, fail_values)
    return qc.to_gate(label=' sum_mutant2 ')

def get_sum_mutant_gate(nbits: int,
                        add_sum_bits: Callable[[QuantumCircuit,
                                                Union[int, Qubit],
                                                Union[int, Qubit],
                                                Union[int, Qubit],
                                                Union[int, Qubit]], None]) \
                                                -> Gate:
    qsum = QuantumCircuit(3*nbits)
    add_sum_mutant1_nbits(qsum, nbits, add_sum_bits, barrier=False)
    sgate = qsum.to_gate(label=' sum_mutant_nbits ')
    return sgate


def view_sum_mutant(nbits: int, sgate: Gate):
    tests = [(n1, n2, res) for n1 in range(2**nbits)
             for n2 in range(2**nbits) for res in [0]]
    for n1, n2, res in tests:
        qc = init_sumcircuit(nbits)
        add_int(qc, n1, 0)
        add_int(qc, n2, nbits)
        qc.append(sgate, range(3*nbits))
        qc.measure(range(2*nbits, 3*nbits), range(nbits))
        qobj = assemble(transpile(qc, backend=sim))
        counts = sim.run(qobj).job.result().get_counts()
        res = int(list(counts)[0], 2)
        print(n1, n2, res, end='...')
        if res == (n1+n2)%(2**nbits):
            print('ok')
        else:
            print(f'error in n1={n1}, n2={n2}, res={res}')



def view_mutants(nbits: int):
    mutants = [
        ('normal function', get_sum_gate(nbits)), # the right function
        ('mutant 1', get_sum_mutant_gate(nbits, add_sum_mutant1_bits)),
        ('mutant 2', get_sum_mutant_gate(nbits, add_sum_mutant2_bits)),
        ('mutant 3', get_sum_mutant2_gate(nbits, fail_values=['1111', '1101'])),
        ('mutant 4', get_sum_mutant2_gate(nbits, fail_values=['1111'])),
    ]
    for mutant_name, mutant  in mutants:
        print(f'----------- {mutant_name} -------------')
        view_mtrule1_gate(nbits, mutant)
        print(f'----------- end {mutant_name} -------------')



'''
    This functions habe been taken from
    https://qiskit.org/textbook/ch-algorithms/grover.html
'''
def diffuser(nqubits: int) -> Gate:
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s


# In[ ]:

def myqft(n, barrier=False, initv=None):
    qc = QuantumCircuit(n)

    if initv is not None:
        qc.initialize(initv)
        qc.barrier()

    for i in reversed(range(n)):
        qc.h(i)
        for j in reversed(range(i)):
            angle = np.pi / (2**(i-j))
            qc.cp(angle, i, j)
        if barrier:
            qc.barrier()

    for i in range(n//2):
        qc.swap(i, n-i-1)
    return qc

def grover_step(nvar_bits: int, ngate_bits: int, oracle: Gate,
                barrier: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(nvar_bits + ngate_bits + 1)
    qc.append(oracle, range(nvar_bits + ngate_bits + 1))
    qc.append(diffuser(nvar_bits), range(nvar_bits))
    return qc


def grover(nvar_bits: int, ngate_bits: int, oracle: Gate,
           steps: int = None, barrier: bool = False,
           save_statevector: bool = True) -> QuantumCircuit:
    '''
    the oracle uses nvar_bits+ngate_bits+1 bits
    the first bits nvar_bits are the input
    the bit in position  nvar_bits is the response of the oracle
    the last ngate_bits are the auxiliary gate of the oracle

    |x> ---nvar_bits--|      |---|x>
    |out> -1----------|oracle|---|out \\oplus f(x)>
    |0> ---ngate_bits-|      |---|0>
    '''
    qc = QuantumCircuit(nvar_bits + ngate_bits + 1, nvar_bits)
    #qc.initialize([1/math.sqrt(2), -1/math.sqrt(2)], nvar_bits)
    qc.x(nvar_bits)
    #qc.h(range(nvar_bits))
    qc.h(range(nvar_bits+1))
    # qc.x(nvar_bits)
    if barrier:
        qc.barrier()
    # quanus = diffuser(nvar_bits)

    #qc.save_statevector(label=f'psi_00')
    if steps is None:
        steps = round(math.pi*math.sqrt(2**nvar_bits)/4)
    for i in range(steps):
        qc.append(grover_step(nvar_bits, ngate_bits, oracle).to_gate(label='O US'),
                  range(nvar_bits+ngate_bits+1))
        # qc.append(oracle, range(nvar_bits + ngate_bits + 1))
        # qc.append(us, range(nvar_bits))
        if barrier:
            qc.barrier()
    if save_statevector:
        qc.save_statevector(label='psi')
    qc.measure(range(nvar_bits), range(nvar_bits))
    return qc



def fake_gate1(nvar_bits: int, ngate_bits: int,
               initial_state: str) -> QuantumCircuit:
    qc = QuantumCircuit(nvar_bits + ngate_bits + 1)
    for i in range(nvar_bits):
        if not int(initial_state[i], 2):
            qc.x(i)
    qc.append(MCXGate(nvar_bits), range(nvar_bits+1))
    for i in range(nvar_bits):
        if not int(initial_state[i], 2):
            qc.x(i)
    return qc



def test_fake1():
    nbits = 3
    initial_state = '101'
    fake = fake_gate1(nbits, 0, initial_state).to_gate()
    sim = Aer.get_backend('aer_simulator')
    for i in range(2**nbits):
        qx = QuantumRegister(nbits, 'x')
        qr = QuantumRegister(1, 'r')
        cr = ClassicalRegister(1, 'cr')
        qc = QuantumCircuit(qx, qr, cr)
        add_int(qc, i, 0)
        qc.append(fake, range(nbits+1))
        qc.measure(qr, cr)
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj, shots=2048)
        counts = list(job.result().get_counts().items())
        print(f'{i:03b}, {counts}', end='...')
        assert len(counts) == 1, f'error {len(counts)}'
        if i == int(initial_state, 2):
            assert counts[0] == ('1', 2048), f'error {counts[0]}'
        else:
            assert counts[0] == ('0', 2048), f'error {counts[0]}'
        print('ok')

def fake_gate2(nvar_bits: int, ngate_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(nvar_bits + ngate_bits + 1)
    qc.cx(0, nvar_bits)
    qc.x(0)
    qc.cx(0, nvar_bits)
    qc.x(0)
    return qc


def test_fake2():
    nbits = 3
    fake = fake_gate2(1, 0).to_gate()
    sim = Aer.get_backend('aer_simulator')
    for i in range(2**nbits):
        qx = QuantumRegister(nbits, 'x')
        qr = QuantumRegister(1, 'r')
        cr = ClassicalRegister(1, 'cr')
        qc = QuantumCircuit(qx, qr, cr)
        add_int(qc, i, 0)
        qc.append(fake, [0, nbits])
        qc.measure(qr, cr)
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj, shots=2048)
        counts = list(job.result().get_counts().items())
        print(f'{i:03b}, {counts}', end='...')
        assert len(counts) == 1, f'error {len(counts)}'
        assert counts[0] == ('1', 2048), f'error {counts[0]}'
        print('ok')


def fake_gate3(nvar_bits:int, noracle_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(nvar_bits + noracle_bits + 1)
    return qc

def test_fake3():
    nbits = 3
    fake = fake_gate3(1, 0).to_gate()
    sim = Aer.get_backend('aer_simulator')
    for i in range(2**nbits):
        qx = QuantumRegister(nbits, 'x')
        qr = QuantumRegister(1, 'r')
        cr = ClassicalRegister(1, 'cr')
        qc = QuantumCircuit(qx, qr, cr)
        add_int(qc, i, 0)
        qc.append(fake, [0, nbits])
        qc.measure(qr, cr)
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj, shots=2048)
        counts = list(job.result().get_counts().items())
        print(f'{i:03b}, {counts}', end='...')
        assert len(counts) == 1, f'error {len(counts)}'
        assert counts[0] == ('0', 2048), f'error {counts[0]}'
        print('ok')


def fake_gate4(nbits: int, nextra: int) -> QuantumCircuit:
    qc = QuantumCircuit(nbits+1+nextra)
    qc.x(0)
    qc.cx(0, nbits)
    qc.x(0)
    return qc

def test_fake4():
    nbits = 4
    fake = fake_gate4(nbits, 0).to_gate()
    sim = Aer.get_backend('aer_simulator')
    for i in range(2**nbits):
        qx = QuantumRegister(nbits, 'x')
        qr = QuantumRegister(1, 'r')
        cr = ClassicalRegister(1, 'cr')
        qc = QuantumCircuit(qx, qr, cr)
        add_int(qc, i, 0)
        qc.append(fake, range(nbits+1))
        qc.measure(qr, cr)
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj, shots=2048)
        counts = list(job.result().get_counts().items())
        print(f'{i:03b}, {counts}', end='...')
        assert len(counts) == 1, f'error {len(counts)}'
        if i % 2 == 0:
            assert counts[0] == ('1', 2048), f'error {counts[0]}'
        else:
            assert counts[0] == ('0', 2048), f'error {counts[0]}'
        print('ok')


def fake_gate5(nbits: int,
               solutions: List[str],
               ngate_bits: int = 0,
               barrier: bool = False) -> QuantumCircuit:
    values = QuantumRegister(nbits, 'values')
    res = QuantumRegister(1, 'res')
    if ngate_bits>0:
        gate_bits = QuantumRegister(ngate_bits, 'gate')
        qc = QuantumCircuit(values, res, gate_bits)
    else:
        qc = QuantumCircuit(values, res)

    for pos, sol in enumerate(solutions):
        sol = sol[::-1]
        for i in range(nbits):
            if int(sol[i], 2) == 0:
                qc.x(values[i])
        qc.append(MCXGate(nbits),
                  values[:] + res[:])
        for i in range(nbits):
            if int(sol[i], 2) == 0:
                qc.x(values[i])
        if barrier:
            qc.barrier()
    return qc

def test_fake5():
    nbits = 5
    solutions = ['11101', '00101']
    fake = fake_gate5(nbits, solutions)
    sim = Aer.get_backend('aer_simulator')
    for i in range(2**nbits):
        qx = QuantumRegister(nbits, 'x')
        qr = QuantumRegister(1, 'r')
        cr = ClassicalRegister(1, 'cr')
        cx = ClassicalRegister(nbits, 'cx')
        qc = QuantumCircuit(qx, qr, cr, cx)
        add_int(qc, i, 0)
        qc.append(fake, range(nbits+1))
        qc.measure(qr, cr)
        qc.measure(qx, cx)
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj, shots=2048)
        counts = list(job.result().get_counts().items())
        print(f'{i:05b}, {counts}', end='...')
        assert len(counts) == 1, f'error {len(counts)}'
        sols = counts[0][0].split()
        assert int(sols[0],2) == i, f'error {i}, {sols[0]}'
        if f'{i:05b}' in solutions:
            print('sol', end='......')
            assert sols[1] == '1', f'error {sols[1]}'
        else:
            assert sols[1] == '0', f'error {sols[0]}'
            print('no sol', end='...')
        print('ok')


def view_vector(vector: np.array, nbits: int):
    for pos, v in filter(lambda x: x[1] != 0,
                         map(lambda x: (x, vector[x]), range(len(vector)))):
        v = 2**(nbits/2) * v
        re = round(v.real, 2)
        im = round(v.imag, 2)
        print(f'{pos:0{nbits}b}: {re}, {im}j')


def simulate_grover(nora_bits: int,
                    naux_bits: int,
                    oracle: Gate,
                    steps=None,
                    shots=1000) -> Dict[str, int]:
    gfake = grover(nora_bits, naux_bits, oracle, steps)
    sim = Aer.get_backend('aer_simulator')
    qobj = assemble(transpile(gfake, backend=sim))
    job = sim.run(qobj, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts


def vector_grover(nora_bits: int, naux_bits:int,
                  oracle: Gate, steps=None) -> None:
    gfake = grover(nora_bits, naux_bits, oracle, steps)
    sim = Aer.get_backend('statevector_simulator')
    qobj = assemble(transpile(gfake, sim))
    job = sim.run(qobj)
    result = job.result()
    data = result.data(0)
    for name, value in sorted(data.items()):
        if name[:3] == 'psi':
            print(f'-------{name}--------')
            view_vector(value, nora_bits+naux_bits+1)
            print('---------------')


def test_simulate_grover_1():
    noracle_bits = 6
    steps = None
    initial_state = '1011'
    nvar_bits = len(initial_state)
    oracle = fake_gate1(nvar_bits, noracle_bits, initial_state).to_gate()
    print('testing grover with fake 1', end='....')
    counts = simulate_grover(nvar_bits, noracle_bits, oracle, steps).items()
    max_value = max(counts, key=lambda x: x[1])[0]
    print(max_value, counts, end='....')
    max_value = max_value[::-1]
    assert max_value == initial_state
    print('ok')


def test_simulate_grover_5():
    noracle_bits = 0
    steps = None
    solutions = ['10011', '01100']
    nvar_bits = len(solutions[0])
    oracle = fake_gate5(nvar_bits, solutions).to_gate()
    print('testing grover with fake 5', end='....')
    counts = simulate_grover(nvar_bits, noracle_bits, oracle, steps).items()
    max_values = sorted(counts, key=lambda x: x[1], reverse=True)[:2]
    print(max_values, counts, end='....')
    #assert max_value == initial_state
    #print('ok')
    print()


def test_vector_grover_1():
    nvar_bits = 3
    noracle_bits = 6
    steps = None
    oracle = fake_gate1(nvar_bits, noracle_bits, '101').to_gate()
    vector_grover(nvar_bits, noracle_bits, oracle, steps)



def plt_grover_all(nora_bits:int, naux_bits:int,
                   oracle: Gate) -> None:
    max_steps = math.ceil(math.pi*math.sqrt(2**nora_bits))
    fig, axs = plt.subplots(nrows=max_steps-1, figsize=(3, 3*max_steps))
    for steps in range(1, max_steps):
        gfake = grover(nora_bits, naux_bits, oracle, steps)
        sim = Aer.get_backend('aer_simulator')
        qobj = assemble(transpile(gfake, backend=sim))
        job = sim.run(qobj, shots=2048)
        counts = job.result().get_counts()
        plot_histogram(counts, ax=axs[steps-1])
    fig.show()


def grover_file(nora_bits: int, naux_bits: int,
                oracle: Gate, fout: TextIOBase) -> None:
    max_steps = math.ceil(math.pi*math.sqrt(2**nora_bits))
    for steps in range(1, max_steps):
        print(f'pasos: {steps}', end='...')
        gfake = grover(nora_bits, naux_bits, oracle, steps)
        sim = Aer.get_backend('aer_simulator')
        qobj = assemble(transpile(gfake, backend=sim))
        job = sim.run(qobj, shots=2048)
        counts = job.result().get_counts()
        fout.write(f'{steps}:{counts}\n')
        print('ok')



def plt_mtrule1(nbits: int, sumgate: Gate,
                 steps: int = None) -> None:
    qc = grover(nbits, 2*nbits+1,
            get_mtrule1_gate(nbits, sumgate),
            steps=steps, barrier=True)
    sim = Aer.get_backend('aer_simulator')
    qobj = assemble(transpile(qc, backend=sim))
    job = sim.run(qobj, shots=2048)
    counts = job.result().get_counts()
    print(counts)
    fig, ax = plt.subplots()
    plot_histogram(counts, ax=ax)
    fig.show()



def test2_mtrule1(sumgate: Gate, init_state: str):
    nbits = len(init_state)
    qc = QuantumCircuit(3*nbits+2, nbits+1)
    for i in range(nbits):
        if int(init_state[i], 2):
            qc.x(i)
    qc.initialize([1/math.sqrt(2), -1/math.sqrt(2)], nbits)
    qc.measure(range(nbits), range(nbits))
    mtrule = get_mtrule1_gate(nbits, sumgate)
    backend = BasicAer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    psi  = result.get_statevector(qc)
    return qc, psi


def view_grover_mtrule1(nbits: int, sumgate: Gate,
                        steps: int) -> None:
    mtrule1_circ = mtrule1(nbits, sumgate)
    qc = grover(nbits, 2*nbits+1,
                mtrule1_circ.to_gate(),
                steps=steps, barrier=True)
    print(qc.draw(fold=-1))


def test_mtrule1_grover(nbits: None, sumgate: Gate,
                        steps: int = None) -> None:
    mtrule1_circ = mtrule1(nbits, sumgate, barrier=False)
    qc = grover(nbits, 2*nbits+1,
                mtrule1_circ.to_gate(),
                steps=steps, barrier=True)
    sim = Aer.get_backend('aer_simulator')
    qobj = assemble(transpile(qc, backend=sim))
    job = sim.run(qobj, shots=2048)
    counts = job.result().get_counts()
    return counts


def mtrule1_grover_vector(nbits, sumgate, steps=None):
    mtrule1_circ = mtrule1(nbits, sumgate)
    qc = grover(nbits, 2*nbits+1,
                mtrule1_circ.to_gate(),
                steps=steps, barrier=True)
    sim = Aer.get_backend('statevector_simulator')
    qobj = assemble(transpile(qc, backend=sim))
    result = sim.run(qobj).result()
    return result.data(0)


def view_grover_mtrule1_extended(nbits: int, sumgate: Gate,
                                 steps: int) -> None:
    """
    We use an extended version of the gate such that the search
    space is duplicated.
    """
    mtrule1_circ = mtrule1(nbits, sumgate)
    ecirc = extended_oracle(mtrule1_circ)
    qc = grover(nbits+1, 2*nbits+1,
                ecirc,
                steps=steps, barrier=True)
    print(qc.draw(fold=-1))


def test_grover_mtrule1_extended(nbits: None, sumgate: Gate,
                                 steps: int = None) -> None:
    """
    We use an extended version of the gate such that the search
    space is duplicated.
    """
    mtrule1_circ = mtrule1(nbits, sumgate, barrier=False)
    ecirc = extended_oracle(mtrule1_circ)
    qc = grover(nbits+1, 2*nbits+1,
                ecirc,
                steps=steps, barrier=True)
    sim = Aer.get_backend('aer_simulator')
    qobj = assemble(transpile(qc, backend=sim))
    job = sim.run(qobj, shots=2048)
    counts = job.result().get_counts()
    return counts


def set_reg(qc, n, reg):
    qbits = reg[:]
    for i in qbits:
        if n%2 == 1:
            qc.x(i)
        n = n//2

def mtrule2(size: int, sgate: Gate,
            x: Union[None, int] = None,
            y: Union[None, int] = None,
            barrier: bool = False) -> QuantumCircuit:
    xreg = QuantumRegister(size, 'x')
    yreg = QuantumRegister(size, 'y')
    one = QuantumRegister(size, 'one')
    res1 = QuantumRegister(size, 'res1')
    res2 = QuantumRegister(size, 'res2')
    res3 = QuantumRegister(size, 'res3')
    res4 = QuantumRegister(size, 'res4')
    anc = QuantumRegister(1, 'anc')
    res = QuantumRegister(1, 'res')
    qc = QuantumCircuit(xreg, yreg, res, one,
                        res1, res2, res3, res4, anc)
    if x is not None:
        set_reg(qc, x, xreg)
    if y is not None:
        set_reg(qc, y, yreg)
    if barrier:
        qc.barrier()
    set_reg(qc, 1, one)
    eqgate = eq_gate(size)
    qc.append(sgate, yreg[:]+one[:]+res1[:])
    qc.append(sgate, xreg[:]+res1[:]+res2[:])
    qc.append(sgate, xreg[:]+yreg[:]+res3[:])
    qc.append(sgate, one[:]+res3[:]+res4[:])
    qc.append(eqgate, anc[:]+res2[:]+res4[:]+res[:])
    qc.append(sgate.inverse(), one[:]+res3[:]+res4[:])
    qc.append(sgate.inverse(), xreg[:]+yreg[:]+res3[:])
    qc.append(sgate.inverse(), xreg[:]+res1[:]+res2[:])
    qc.append(sgate.inverse(), yreg[:]+one[:]+res1[:])
    if barrier:
        qc.barrier()
    qc.x(res)
    return qc


def view_mtrule2_gate(size: int, sgate: Gate) -> None:
    tests = [(x, y) for x in range(2**size) for y in range(2**size)]
    sim = Aer.get_backend('aer_simulator')
    ok, fail = 0, 0
    for x, y in tests:
        qc = mtrule2(size, sgate, x, y)
        xreg, yreg, res, one, res1, res2, res3, res4, anc = qc.qregs
        cxreg = ClassicalRegister(size, 'cx')
        cyreg = ClassicalRegister(size, 'cy')
        cone = ClassicalRegister(size, 'cone')
        cres1 = ClassicalRegister(size, 'cres1')
        cres2 = ClassicalRegister(size, 'cres2')
        cres3 = ClassicalRegister(size, 'cres3')
        cres4 = ClassicalRegister(size, 'cres4')
        canc = ClassicalRegister(1, 'canc')
        cres = ClassicalRegister(1, 'cres')
        qc.add_register(cxreg, cyreg, cone,
                        cres1, cres2, cres3, cres4,
                        canc, cres)
        qc.measure(xreg[:], cxreg[:])
        qc.measure(yreg[:], cyreg[:])
        qc.measure(one[:], cone[:])
        qc.measure(res1[:], cres1[:])
        qc.measure(res2[:], cres2[:])
        qc.measure(res3[:], cres3[:])
        qc.measure(res4[:], cres4[:])
        qc.measure(anc[:], canc[:])
        qc.measure(res[:], cres[:])
        qobj = assemble(transpile(qc, backend=sim))
        job = sim.run(qobj)
        counts = list(job.result().get_counts())
        print(f'{x:04b}, {y:04b}, {counts}', end='...')
        values = list(map(lambda x: int(x, 2), counts[0].split()))
        values.reverse()
        # print(values, end='...')
        xs, ys, ones, res1s, res2s, res3s, res4s, ancs, ress = values
        print(ys, ones, res1s, end='.')
        assert res1s == 0
        print(res2s, end='.')
        assert res2s == 0
        print(res3s, end='.')
        assert res3s == 0
        print(res4s, end='.')
        assert res4s == 0
        print(ress, end='.')
        # assert values[:3] == [0, 0, 0], f'error values[:3]={values[:3]}'
        # assert values[4] == t, f'error values[4]={values[4]}'
        if ress == 0:
            print('ok')
            ok += 1
        else:
            print('fail')
            fail += 1
    print(f'{fail}/{2**(2*size)}')


def view_grover_mtrule2(nbits: int, sumgate: Gate,
                        steps: int) -> None:
    mtrule2_circ = mtrule2(nbits, sumgate)
    qc = grover(2*nbits, 5*nbits+1,
                mtrule2_circ.to_gate(),
                steps=steps, barrier=True)
    print(qc.draw(fold=-1))

def test_mtrule2_grover(nbits: None, sumgate: Gate,
                        steps: int = None) -> None:
    mtrule2_circ = mtrule2(nbits, sumgate)
    qc = grover(2*nbits, 5*nbits+1,
                mtrule2_circ.to_gate(),
                steps=steps, barrier=True)
    sim = Aer.get_backend('aer_simulator')
    qobj = assemble(transpile(qc, backend=sim))
    job = sim.run(qobj, shots=2048)
    counts = job.result().get_counts()
    return counts


def mtrule2_grover_vector(nbits, sumgate, steps=None):
    mtrule2_circ = mtrule2(nbits, sumgate)
    qc = grover(2*nbits, 5*nbits+1,
                mtrule2_circ.to_gate(),
                steps=steps, barrier=True)
    sim = Aer.get_backend('statevector_simulator')
    qobj = assemble(transpile(qc, backend=sim))
    result = sim.run(qobj).result()
    return result.data(0)


def test_grover_file(nbits: int) -> None:
    n = 2
    oracle = mtrule2(nbits, get_sum_mutant2_gate(nbits, fail_values=['10'])).to_gate()
    with open(f'mtrule2_{nbits}', 'w') as fout:
        grover_file(2*nbits, 5*n+1, oracle, fout)



def test_mtrule2_grover_vector(nbits: int, sgate: Gate) -> Gate:
    mtrule2_circ = mtrule2(nbits, sgate)
    vector_grover(2*nbits, 5*nbits+1, mtrule2_circ.to_gate())



def QPE(circuit, nbits_est, vector=None, measure=True):
    qest = QuantumRegister(nbits_est, 'est')
    state = QuantumRegister(circuit.num_qubits, 'st')
    cest = ClassicalRegister(nbits_est, 'cest')
    qpe = QuantumCircuit(qest, state, cest)

    pos_state = nbits_est
    if vector is None:
        qpe.h(state)
    else:
        for i in range(circuit.num_qubits):
            if type(vector) is str:
                if vector[i] == "1":
                    qpe.x(pos_state+i)
                elif vector[i] == "h":
                    qpe.h(pos_state+i)
            elif type(vector) is list:
                qpe.initialize(vector, state[:])

    qpe.h(qest)
    repetitions = 1
    for counting_qubit in qest:
        for i in range(repetitions):
            qpe.append(circuit.control(1, label=' CU '), [counting_qubit]+state[:]);
        repetitions *= 2
    qpe.barrier()
    # Apply inverse QFT
    qpe.append(myqft(nbits_est).to_gate(label='QFT').inverse(), qest)
    #iqft = QFT(nbits_est, inverse=True, do_swaps=False).reverse_bits()
    #qpe.compose(iqft, qest[:], inplace=True)
    #
    # Measure
    qpe.barrier()
    if measure:
        qpe.measure(qest, cest)
    return qpe

def test_QPE():
    qc = QuantumCircuit(1)
    qc.p(2*np.pi*0.75, 0)
    #qc.h(0)
    qpe = QPE(qc, 2, vector=[0,1], measure=True)
    counts = execute(qpe, backend=AER_SIM).result().get_counts()
    print(counts)


def quantum_count(nvar_bits: int, ngate_bits: int, nest_bits: int,
                  oracle: Gate,
                  barrier: bool = False) -> QuantumCircuit:
    '''
    the oracle uses nvar_bits+ngate_bits+1 bits
    the first bits nvar_bits are the input
    the bit in position  nvar_bits is the response of the oracle
    the last ngate_bits are the auxiliary gate of the oracle

    |x> ---nvar_bits--|      |---|x>
    |out> -1----------|oracle|---|out \\oplus f(x)>
    |0> ---ngate_bits-|      |---|0>
    '''
    qest = QuantumRegister(nest_bits, 'est')
    qgrov = QuantumRegister(nvar_bits+ngate_bits+1, 'grover')
    res = ClassicalRegister(nest_bits, 'res')
    qc = QuantumCircuit(qest, qgrov, res)
    qc.h(qest)
    qc.x(nest_bits+nvar_bits)
    qc.h(range(nest_bits, nest_bits+nvar_bits+1))

    print(qc.draw())
    gs = grover_step(nvar_bits, ngate_bits, oracle)
    # print(gs.draw(fold=-1))
    # qc=QPE(gs.to_gate(label=" G "), nest_bits, vector="", measure=True)
    # print(qc.draw(fold=-1))
    power = 1
    for qbit in qest:
        for _ in range(power):
            qc.append(gs.to_gate(label='G').control(), [qbit] + qgrov[:])
        power *= 2
        if barrier:
            qc.barcrier()
    qc.append(myqft(nest_bits).inverse(), qest)
    qc.measure(qest, res)
    print(qc.draw(fold=-1))
    return qc


def test_qutantum_count(nbits: int, nest: int, oracle: Gate,
                        ngate_bits: int = 0) -> None:
    # fake = fake_gate1(nbits, 0, initial_state='001').to_gate()
    qc = quantum_count(nbits, ngate_bits, nest, oracle)
    sim = Aer.get_backend('aer_simulator')
    counts = execute(qc, backend=sim).result().get_counts().items()
    counts = sorted(counts, key=lambda x: x[1], reverse=True)
    print(counts)
    theta = (2*np.pi)*(2**-nest)*int(counts[0][0], 2)
    N = 2**nbits
    M = N*np.cos(theta/2)**2  # It is the cos instead the the sin because
    # the diffuser implementation.
    # https://qiskit.org/textbook/ch-algorithms/quantum-counting.html
    err = (math.sqrt(2*M*N) + N/(2**(nest+1)))*(2**(-nest))
    print(f'number of solutions: {M}')
    print(f'error: {err}')
    return counts


def test_qutantum_count_fake5_3() -> None:
    nbits = 4
    nest = nbits
    solutions = ['1100', '1010', '0000']
    # nbits = 3
    fake = fake_gate5(nbits, solutions, barrier=False)
    print(fake.draw(fold=-1))
    test_qutantum_count(nbits, nest, fake.to_gate(label=' F5 '))


def test_qutantum_count_fake5_1() -> None:
    nbits = 4
    nest = nbits
    solutions = ['1100']
    # nbits = 3
    fake = fake_gate5(nbits, solutions, barrier=False)
    print(fake.draw(fold=-1))
    test_qutantum_count(nbits, nest, fake.to_gate(label=' F5 '))


def test_qutantum_count_fake1() -> None:
    nbits = 4
    nest = nbits
    init_state = '1100'
    # nbits = 3
    fake = fake_gate1(nbits, 0, init_state)
    print(fake.draw(fold=-1))
    test_qutantum_count(nbits, nest, fake.to_gate(label=' F1 '))


def test_qutantum_count_fake3() -> None:
    nbits = 4
    nest = nbits//2
    # nbits = 3
    fake = fake_gate3(nbits, 0)
    print(fake.draw(fold=-1))
    test_qutantum_count(nbits, nest, fake.to_gate(label=' F3 '))


def test_qutantum_count_fake2() -> None:
    nbits = 4
    nest = nbits//2
    fake = fake_gate2(nbits, 0)
    print(fake.draw(fold=-1))
    test_qutantum_count(nbits, nest, fake.to_gate(label=' F2 '))



if __name__ == '__main__':
    nbits = int(sys.argv[1])
    sgate = get_sum_mutant2_gate(nbits, fail_values=['101'])
    test_mtrule2_gate(nbits, sgate)
    test_mtrule2_grover_vector(nbits, sgate)
