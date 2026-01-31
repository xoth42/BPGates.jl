@testitem "Two one-sec/one two-sec gate equivalent" begin
# Test that applying two noise gates of 1 second duration has the same effect as applying one noise gate of two seconds 
# Test for T1, T2 Noise ops. Helper function 'test_gate_equiv' will 

using Test
using BPGates
using QuantumClifford: mctrajectories
using BPGates: T1NoiseOp, T2NoiseOp, BellState, BellMeasure

N = 10000
DEBUG = true
t_decay = 100
lambda_func(t)  = 1 - exp(-t/t_decay)
# times = [0.001, 1,10,100,10000, 100000]  # given decay time = 100, this tests from small jump chance (0.01 decay time) to large jump chance (1000 decay time)
times = [100]
"""Helper function to run the one/two sec test given gate generator and measure.

    gatefunction := (t) -> some BPGates op
"""
function test_gate_equiv(gate_function,measure=BellMeasure(1,1))
    for time in times
        if DEBUG
            println("time=$time")
        end
        one_sec = [gate_function(time), gate_function(time), measure]
        two_sec = [gate_function(2*time),                    measure]

        for b1 in 1:4
            for b2 in 1:4
                s1 = BPGates.int_to_bit(b1, Val(2))
                s2 = BPGates.int_to_bit(b2, Val(2))
                s = (s1..., s2...) # concatenate tuples to get 4 bits

                ones_result = mctrajectories(BellState(s), one_sec;trajectories=N)
                two_result = mctrajectories(BellState(s), two_sec;trajectories=N)

                # avg results
                ones_result = Dict(k => v/N for (k,v) in ones_result)
                two_result = Dict(k => v/N for (k,v) in two_result)

                if DEBUG
                    print("Gate: $(gate_function), time=$time, B1=$b1, B2=$b2, ")
                    println("One-sec / two-sec results:")
                    for (status, count) in ones_result
                        println("  Status: $status, One-sec Count: $count, Two-sec Count: $(two_result[status])")
                    end
                end
                
                for status in keys(ones_result)
                    @test isapprox(ones_result[status], two_result[status]; atol=10/sqrt(N))
                end
            end
        end
    end
end

@testset "T1 noise" begin
    T1_gate_function(t) = T1NoiseOp(1, lambda_func(t))
    measures = [BellMeasure(i,1) for i in 1:3] # x y z measure
    for measure in measures
        test_gate_equiv(T1_gate_function, measure)
    end
end

@testset "T2 noise" begin
    # ignoring the t1 effect on the lambda function (t1 = ∞), so it simplifies to just the same exp decay.
    T2_gate_function(t) = T2NoiseOp(1, lambda_func(t))
    measures = [BellMeasure(i,1) for i in 1:3] # x y z measure
    for measure in measures
        test_gate_equiv(T2_gate_function, measure)
    end
end

end
