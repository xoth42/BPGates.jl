@testitem "Two one-sec/one two-sec gate equivalent" tags=[:noise] begin
# Test that applying two noise gates of 1 second duration has the same effect as applying one noise gate of two seconds 
# Test for T1, T2 Noise ops. Helper function 'test_gate_equiv' will 
using Revise
using Test
using BPGates
using QuantumClifford: mctrajectories
using BPGates: T1NoiseOp, T2NoiseOp, BellState, BellMeasure, ADCOp,T1DepolarizingOp, ManualADCOp

N = 10000
DEBUG = false
t_decay = 100
lambda_func(t)  = 1 - exp(-t/t_decay)
# times = [0.001, 1,10,100,10000, 100000]  # given decay time = 100, this tests from small jump chance (0.01 decay time) to large jump chance (1000 decay time)
times = [.1,1,10,100,1000]
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


@testset "ADCOp noise" begin
    t1s = [10,100,300,10,300]
    t2s = [15,140,400,300,10]
    # t2 > t1 case has errors
    for i in eachindex(t1s)
        print("Testing ADCOp with T1=$(t1s[i]), T2=$(t2s[i])\n")
        T1 = t1s[i]
        T2 = t2s[i]
        px_py(t) = (1/4)*(1-exp(-t/T1))
        pz(t) = (1/2)*(1-exp(-t/T2))-(1/4)*(1-exp(-t/T1))
        func(t) = ADCOp(1,px_py(t),pz(t))
        measures = [BellMeasure(i,1) for i in 1:3] # x y z measure
        for measure in measures
            test_gate_equiv(func, measure)
        end
    end
end


@testset "ManualADCOp noise" begin
    t1s = [10,100,300,10,300]
    t2s = [15,140,400,300,10]
    # t2 > t1 case has errors
    for i in eachindex(t1s)
        # debug &&
         print("Testing ManualADCOp with T1=$(t1s[i]), T2=$(t2s[i])\n")
        T1 = t1s[i]
        T2 = t2s[i]
        px_py(t) = (1/4)*(1-exp(-t/T1))
        pz(t) = (1/2)*(1-exp(-t/T2))-(1/4)*(1-exp(-t/T1))
        func(t) = ManualADCOp(1,px_py(t),pz(t))
        measures = [BellMeasure(i,1) for i in 1:3] # x y z measure
        for measure in measures
            test_gate_equiv(func, measure)
        end
    end
end



@testset "T1DepolarizingOp noise" begin
    T1Depol_gate_function(t) = T1DepolarizingOp(1, lambda_func(t))
    measures = [BellMeasure(i,1) for i in 1:3] # x y z measure
    for measure in measures
        test_gate_equiv(T1Depol_gate_function, measure)
    end
end


# end


# using Test
using BPGates
using QuantumClifford: mctrajectories

function find_faulty_adc_region(; t1_range=10:10:300, t2_range=200:10:400)
    faulty_cases = []
    for T1 in t1_range
        found_faulty = false
        for T2 in t2_range
            if T2 < T1 || found_faulty
                continue
            end
            px_py(t) = (1/4)*(1-exp(-t/T1))
            pz(t) = (1/2)*(1-exp(-t/T2))-(1/4)*(1-exp(-t/T1))
            func(t) = ADCOp(1,px_py(t),pz(t))
            measures = [BellMeasure(i,1) for i in 1:3]
            try
                for measure in measures
                    test_gate_equiv(func, measure)
                end
            catch e
                println("Failure at T1=$T1, T2=$T2: $e")
                push!(faulty_cases, (T1, T2))
                found_faulty = true
                break  # Stop testing higher T2 for this T1
            end
        end
    end
    return faulty_cases
end
# faulty = find_faulty_adc_region()
# println("Faulty (T1, T2) pairs: ", faulty)

end