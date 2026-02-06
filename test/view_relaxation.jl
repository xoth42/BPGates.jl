using BPGates: T1NoiseOp, T2NoiseOp, T1DepolarizingOp, ADCOp, BellState, ManualADCOp
using QuantumClifford: mctrajectories, apply!
using BPGates
using GLMakie
using OhMyThreads: tmap, tmapreduce, index_chunks
using Base.Threads: nthreads
using OhMyThreads
# ============================================================================
# Configuration
# ============================================================================
SIMS = 2000000
T_DECAY = 100  # decay time constant

# Lambda functions for different noise types
λ₁_func(t, t1) = 1 - exp(-t / t1)
λ₂_func(t, t1, t2) = 1 - exp(-t/t2 + t/(2t1))

# Status keys for plotting
STATUS_KEYS = [:continue, :failure, :true_success, :false_success]
STATUS_COLORS = Dict(
    :continue => :blue,
    :failure => :red,
    :true_success => :green,
    :false_success => :orange
)

# ============================================================================
# Helper Functions
# ============================================================================

"""Create a BellState from two Bell pair indices (1-4 each)."""
function get_bell(b1::Int, b2::Int)
    s1 = BPGates.int_to_bit(b1, Val(2))
    s2 = BPGates.int_to_bit(b2, Val(2))
    return BellState((s1..., s2...))
end

"""Run simulation and return normalized results."""
function get_sim_results(bell::BellState, gates; traj=SIMS)
    res = mctrajectories(bell, gates; trajectories=traj)
    return Dict(k => v / traj for (k, v) in res)
end

"""Extract a specific status probability from results dict."""
function get_status_prob(results::Dict, status_symbol::Symbol)
    for (k, v) in results
        if contains(string(k), string(status_symbol))
            return v
        end
    end
    return 0.0
end

"""
    Collect final state probabilities over a range of times for single-gate application.
    Applies noise gates WITHOUT measurement to see direct effect on state.

    Returns Dict{String, Vector{Float64}} with probabilities for each final state over time.

    State labeling from BPGates notation table:
    Bit representation: (bit1, bit2) where
    - bit2 = XX phase (0:+ →Φ, 1:- → Ψ)
    - bit1 = ZZ phase (0:+ → ⁺, 1:- → ⁻)

    - "00" (XX+, ZZ+) → Φ⁺ (blue)
    - "10" (XX+, ZZ-) → Φ⁻ (red)
    - "01" (XX-, ZZ+) → Ψ⁺ (green)
    - "11" (XX-, ZZ-) → Ψ⁻ (orange)
"""
function single_gate_sweep(times, op_func, lambda_func, bell::BellState)
    final_states = Dict(
        "00" => Float64[],
        "10" => Float64[],
        "01" => Float64[],
        "11" => Float64[]
    )
    
    for t in times
        λ = lambda_func(t)
        gate = [op_func(1, λ)]
        
        # Run many trajectories and track final states
        state_counts = Dict("00" => 0, "01" => 0, "10" => 0, "11" => 0)
        for _ in 1:SIMS
            s = copy(bell)
            for g in gate
                apply!(s, g)
            end
            # Extract final state from first Bell pair (bits 1-2)
            bit1 = Int(s.phases[1])
            bit2 = Int(s.phases[2])
            final_bits = "$(bit1)$(bit2)"
            state_counts[final_bits] += 1
        end
        
        # Normalize to probabilities
        for state in keys(final_states)
            push!(final_states[state], state_counts[state] / SIMS)
        end
    end
    return final_states
end

"""
Collect final state probabilities for cumulative gate application (many small steps).

Returns Dict{String, Vector{Float64}} with probabilities for each final state over time.
"""
function many_gate_sweep(dt, total_time, op_func, lambda_func, bell::BellState)
    final_states = Dict(
        "00" => Float64[],
        "10" => Float64[],
        "01" => Float64[],
        "11" => Float64[]
    )
    
    n_steps = Int(ceil(total_time / dt))
    λ = lambda_func(dt)
    gates = []
    
    for step in 0:n_steps
        # Track state at this step
        state_counts = Dict("00" => 0, "01" => 0, "10" => 0, "11" => 0)
        
        if step > 0
            push!(gates, op_func(1, λ))
        end
        
        for _ in 1:SIMS
            s = copy(bell)
            for g in gates
                apply!(s, g)
            end
            # Extract final state from first Bell pair (bits 1-2)
            bit1 = Int(s.phases[1])
            bit2 = Int(s.phases[2])
            final_bits = "$(bit1)$(bit2)"
            state_counts[final_bits] += 1
        end
        
        # Normalize to probabilities and store
        for state in keys(final_states)
            push!(final_states[state], state_counts[state] / SIMS)
        end
    end
    return final_states
end

"""Plot all final state probabilities on an axis."""
function plot_status_results!(ax, times, results::Dict; label_prefix="", linestyle=:solid)
    state_colors = Dict(
        "00" => :blue,
        "10" => :red,
        "01" => :green,
        "11" => :orange
    )
    
    state_labels = Dict(
        "00" => "Φ⁺",
        "10" => "Φ⁻",
        "01" => "Ψ⁺",
        "11" => "Ψ⁻"
    )
    
    for state in ["00", "01", "10", "11"]
        if haskey(results, state)
            lines!(ax, times, results[state]; 
                   color=state_colors[state], 
                   linestyle=linestyle,
                   label="$(label_prefix) |$(state_labels[state])⟩")
        end
    end
end

# ============================================================================
# Generalized Noise Visualization
# ============================================================================

function plot_noise_results(op_func, lambda_func, op_name, title_name; 
                           t_values=T_DECAY, times=0:10:2000, bell=get_bell(1, 1), many_states=true)
    fig = Figure(size=(1400, 900))
    ax = Axis(fig[1, 1]; 
              xlabel="Time (μs)", 
              ylabel="Probability",
              title="$(title_name)")
    
    # Single gate sweep
    single_results = single_gate_sweep(times, op_func, lambda_func, bell)
    plot_status_results!(ax, collect(times), single_results; label_prefix="single")
    
    # Many gate sweep with different dt values
    if many_states
        # for dt in [10, 50]
        for dt in [50]
            many_results = many_gate_sweep(dt, maximum(times), op_func, lambda_func, bell)
            time_points = 0:dt:maximum(times)
            plot_status_results!(ax, collect(time_points), many_results; 
                                label_prefix="dt=$dt", linestyle=:dash)
        end
    end
    
    axislegend(ax; position=:rt)
    display(fig)
    return fig
end

# ============================================================================
# T1 Noise Visualization
# ============================================================================

function plot_T1_results(; t1=T_DECAY, times=0:10:2000, bell=get_bell(1, 1))
    lambda_func = t -> λ₁_func(t, t1)
    return plot_noise_results(T1NoiseOp, lambda_func, "T1NoiseOp", 
                             "T1 Noise: |Φ⁺⟩ (T1=$(t1) μs)"; 
                             t_values=t1, times=times, bell=bell)
end

# ============================================================================
# T2 Noise Visualization  
# ============================================================================

function plot_T2_results(; t1=T_DECAY, t2=150, times=0:10:2000, bell=get_bell(1, 1))
    lambda_func = t -> λ₂_func(t, t1, t2)
    return plot_noise_results(T2NoiseOp, lambda_func, "T2NoiseOp", 
                             "T2 Noise: |Φ⁺⟩ (T1=$(t1) μs, T2=$(t2) μs)"; 
                             t_values=(t1, t2), times=times, bell=bell)
end

# ============================================================================
# ADC (Amplitude Damping Channel) Noise Visualization
# ============================================================================

function plot_ADC_results(; T1=100, T2=140, times=0:50:1500, bell=get_bell(1, 1))
    # ADCOp parameters based on amplitude damping + dephasing model
    px_py_func(t) = (1/4)*(1-exp(-t/T1))
    pz_func(t) = (1/2)*(1-exp(-t/T2)) - (1/4)*(1-exp(-t/T1))
    lambda_func = t -> (px_py_func(t), pz_func(t))  # Return tuple of probabilities
    
    # Wrapper to create ADCOp with current time parameters
    adc_op_func = (idx, params) -> begin
        px_py, pz = params
        ADCOp(idx, px_py, pz)
    end
    
    return plot_noise_results(adc_op_func, lambda_func, "ADCOp", 
                             "ADC (Amplitude Damping + Dephasing): |Φ⁺⟩ (T1=$(T1), T2=$(T2))"; 
                             t_values=(T1, T2), times=times, bell=bell)
end


function plot_ManualADC_results(; T1=100, T2=140, times=0:50:1500, bell=get_bell(1, 1))
    # ADCOp parameters based on amplitude damping + dephasing model
    px_py_func(t) = (1/4)*(1-exp(-t/T1))
    pz_func(t) = (1/2)*(1-exp(-t/T2)) - (1/4)*(1-exp(-t/T1))
    lambda_func = t -> (px_py_func(t), pz_func(t))  # Return tuple of probabilities
    
    # Wrapper to create ADCOp with current time parameters
    adc_op_func = (idx, params) -> begin
        px_py, pz = params
        ManualADCOp(idx, px_py, pz)
    end
    
    return plot_noise_results(adc_op_func, lambda_func, "ManualADCOp", 
                             "ManualADC (Amplitude Damping + Dephasing): |Φ⁺⟩ (T1=$(T1), T2=$(T2))"; 
                             t_values=(T1, T2), times=times, bell=bell)
end


# ============================================================================
# T1DepolarizingOp Noise Visualization
# ============================================================================

function plot_T1Depolarizing_results(; t1=T_DECAY, times=0:10:2000, bell=get_bell(1, 1))
    lambda_func = t -> λ₁_func(t, t1)
    return plot_noise_results(T1DepolarizingOp, lambda_func, "T1DepolarizingOp", 
                             "T1DepolarizingOp Noise: |Φ⁺⟩ (T1=$(t1) μs)"; 
                             t_values=t1, times=times, bell=bell)
end

# ============================================================================
# Compare all noise types side by side
# ============================================================================

function plot_all_noise_comparison(; t1=T_DECAY, t2=150, T1_adc=100, T2_adc=140, times=0:10:2000, bell=get_bell(1, 1), many_states=true)
    fig = Figure(size=(1800, 1200))
    
    # ADC parameters
    px_py_func(t) = (1/4)*(1-exp(-t/T1_adc))
    pz_func(t) = (1/2)*(1-exp(-t/T2_adc)) - (1/4)*(1-exp(-t/T1_adc))
    adc_lambda_func = t -> (px_py_func(t), pz_func(t))
    adc_op_func = (idx, params) -> begin
        px_py, pz = params
        ADCOp(idx, px_py, pz)
    end
    manual_adc_op_func = (idx, params) -> begin
        px_py, pz = params
        ManualADCOp(idx, px_py, pz)
    end
    noise_ops = [
        (T1NoiseOp, t -> λ₁_func(t, t1), "T1NoiseOp", "T1 (Energy Relaxation)"),
        (T2NoiseOp, t -> λ₂_func(t, t1, t2), "T2NoiseOp", "T2 (Dephasing)"),
        (adc_op_func, adc_lambda_func, "ADCOp", "ADC (Amplitude Damping + Dephasing)"),
        # (T1DepolarizingOp, t -> λ₁_func(t, t1), "T1DepolarizingOp", "T1 Depolarizing")
        (manual_adc_op_func, adc_lambda_func, "ManualADCOp", "Manual ADC (Amplitude Damping + Dephasing)")
    ]
    
    for (idx, (op_func, lambda_func, op_name, title_name)) in enumerate(noise_ops)
        row, col = divrem(idx - 1, 2) .+ 1
        ax = Axis(fig[row, col]; 
                  xlabel="Time (μs)", 
                  ylabel="Probability",
                  title="$(title_name)")
        
        results = single_gate_sweep(times, op_func, lambda_func, bell)
        plot_status_results!(ax, collect(times), results)
        
        if many_states
            many_results = many_gate_sweep(50, maximum(times), op_func, lambda_func, bell)
            time_points = 0:50:maximum(times)
            plot_status_results!(ax, collect(time_points), many_results; 
                                label_prefix="dt=50", linestyle=:dash)
        end
        
        axislegend(ax; position=:rt)
    end
    
    display(fig)
    return fig
end

# ============================================================================
# Compare different initial Bell states
# ============================================================================

function plot_bell_state_comparison(op_func, lambda_func; 
                                    times=0:10:1000, op_name="Noise", many_states=true)
    fig = Figure(size=(1600, 1200))
    
    bell_states = [(1,1), (2,1), (3,1), (4,1)]
    
    for (idx, (b1, b2)) in enumerate(bell_states)
        row, col = divrem(idx - 1, 2) .+ 1
        bell_names = Dict((1,1) => "Φ⁺", (2,1) => "Φ⁻", (3,1) => "Ψ⁺", (4,1) => "Ψ⁻")
        ax = Axis(fig[row, col]; 
                  xlabel="Time (μs)", 
                  ylabel="Probability",
                  title="$(op_name): |$(bell_names[(b1,b2)])⟩")
        
        bell = get_bell(b1, b2)
        results = single_gate_sweep(times, op_func, lambda_func, bell)
        plot_status_results!(ax, collect(times), results)
        
        if many_states
            many_results = many_gate_sweep(50, maximum(times), op_func, lambda_func, bell)
            time_points = 0:50:maximum(times)
            plot_status_results!(ax, collect(time_points), many_results; 
                                label_prefix="dt=50", linestyle=:dash)
        end
        
        axislegend(ax; position=:rt)
    end
    
    display(fig)
    return fig
end


# Relaxation inconsistency - given a range of params - (t1,t2) plot the difference between results via 1-gate and results via 2-gate
# since we have a plot with x=t1 and y=t2, we will plot the error as a color. 


function apply_gate_keep_states(gates, state, n=SIMS)
    final_states = Dict(
        "00" => Float64[],
        "10" => Float64[],
        "01" => Float64[],
        "11" => Float64[]
    )
    
    # Process iterations in parallel chunks - each thread gets its own local counter
    results = tmap(index_chunks(1:n; n=nthreads())) do chunk
        local_counts = Dict("00" => 0, "10" => 0, "01" => 0, "11" => 0)
        for _ in chunk
            s = copy(state)
            for g in gates
                apply!(s, g)
            end
            bit1 = Int(s.phases[1])
            bit2 = Int(s.phases[2])
            final_bits = "$(bit1)$(bit2)"
            local_counts[final_bits] += 1
        end
        return local_counts
    end
    
    # Merge results from all threads
    state_counts = merge(+, results...)
    
    # Normalize to probabilities
    for state in keys(final_states)
        push!(final_states[state], state_counts[state] / n)
    end
    
    return final_states
end


# gate function is of the order gate_function(dt, t1,t2) -> OP
function get_ones_twos(gate_function, time, t1, t2,bellstate)
    one_sec = [gate_function(1*time, t1, t2),gate_function(1*time, t1, t2)]
    two_sec = [gate_function(2*time, t1, t2)]

    ones_result = apply_gate_keep_states(one_sec,bellstate)
    two_result = apply_gate_keep_states(two_sec,bellstate)

    return ones_result, two_result
end


# given state, gate func, measure, time, t1, t2, optional-end-state, make 2d color map plot of inconcistency between one-sec and two-sec
# gate_function(dt, t1,t2) -> OP
function map_2s_inconsistency(gate_function, time, t1s, t2s; bellstate=get_bell(4,1), ax=nothing)
    if ax === nothing
        fig = Figure(size=(800, 600))
        ax = Axis(fig[1,1]; xlabel="T2 (μs)", ylabel="T1 (μs)", title="Inconsistency between 1-sec and 2-sec for $(gate_function) at time=$(time) μs")
    end
    
    # Convert to concrete vectors
    t1s_vec = collect(t1s)
    t2s_vec = collect(t2s)
    
    # Parallelize outer loop over t1 indices using tmapreduce with index_chunks
    inconsistency_rows = tmapreduce(vcat, OhMyThreads.index_chunks(eachindex(t1s_vec); n=nthreads())) do chunk_inds
        local_map = zeros(length(chunk_inds), length(t2s_vec))
        for (local_idx, global_idx) in enumerate(chunk_inds)
            t1 = t1s_vec[global_idx]
            for (j, t2) in enumerate(t2s_vec)
                ones_result, two_result = get_ones_twos(gate_function, time, t1, t2, bellstate)
                
                # calculate inconsistency as sum of absolute differences across all states
                inconsistency = 0.0
                for state in keys(ones_result)
                    inconsistency += abs(ones_result[state][1] - two_result[state][1])
                end
                local_map[local_idx, j] = inconsistency
            end
        end
        return local_map
    end
    
    # inconsistency_rows is already a matrix from vcat
    heatmap!(ax, t2s_vec, t1s_vec, inconsistency_rows; colormap=:viridis)
    return ax

end

# # like map_2s_inconcistency, create a heatmap that shows places where T1 t2 leads to areas of incorrect results. This time, we use an analytical result function (for all 4 populations) and compare one gate application to the theory, summing error in each t1-t2 color point.
# # analyticals(time, t1, t2) -> Dict{String, Float64} of final states
# function map_analytical_inconcistency(gate_function, time, t1s, t2s, analyticals; bellstate=get_bell(4,1), ax=nothing)
#      if ax === nothing
#         fig = Figure(size=(800, 600))
#         ax = Axis(fig[1,1]; xlabel="T2 (μs)", ylabel="T1 (μs)", title="Inconsistency between gate and analytical results for $(gate_function) at time=$(time) μs")
#     end
    
#     inconsistency_map = zeros(length(t1s), length(t2s))
#     for (i,t1) in enumerate(t1s)
#         for (j,t2) in enumerate(t2s)
#             gate_result = apply_gate_keep_states([gate_function(time, t1, t2)], bellstate)
#             analytical_result = analyticals(time, t1, t2)
            
#             # calculate inconsistency as sum of absolute differences across all states
#              inconsistency = 0.0
#              for state in keys(gate_result)
#                 inconsistency += abs(gate_result[state][1] - analytical_result[state][1])
#              end
#              inconsistency_map[i,j] = inconsistency
#         end
#     end

#     # make GLMakie heatmap
#     heatmap!(ax, t2s, t1s, inconsistency_map; colormap=:viridis)
#     return ax
# end


# # define analytical results for relaxation channels
# # result from the decoherence paper 
# qubit_decoherence_density(ρ, t,t1,t2) = begin 
# [
#     1-ρ[2,2]*exp(-t/t1)     ρ[1,2]*exp(-t/t2);
#     ρ[1,2]*exp(-t/(t2))     ρ[2,2]*exp(-t/t1)
# ]
# end



# # Now we use it to make the bell-state result
# # comp -> bell transformation 
# T = [1 1 0 0;
#      0 0 1 1;
#      0 0 1 -1;
#      1 -1 0 0]/sqrt(2)

# # bell_decoherence(bell_state_comp_basis_ket,t, t1, t2) = begin
# #     bell_state_density_comp = bell_state_comp_basis_ket * bell_state_comp_basis_ket'
# #     ρ_comp = kron(qubit_decoherence_density())

# function bell_state_decoherence(bell_vector_comp, t, t1, t2)
#     # get density matrix of bell state in comp
#     ρ = bell_vector_comp * bell_vector_comp'
    
#     # apply decoherence to each qubit
#     ρ_after_1 = qubit_decoherence_density(ρ[1:2,1:2], t, t1, t2)
#     ρ_after_2 = qubit_decoherence_density(ρ[3:4,3:4], t, t1, t2)
    
#     ρ_bell_final = kron(ρ_after_1, ρ_after_2)

    
#     # transform back to Bell basis
#     ρ_bell_final = T' * ρ_bell_final * T
    
#     return ρ_bell_final
# end

# bell_states = [T[:,1], T[:,2], T[:,3], T[:,4]]


# # Test for a state
# b = bell_states[1]
# # initial density 
# b * b'
# bell_state_decoherence(b, 0, 100, 150)

# bell_state_decoherence(b, 1, 100, 150)
# bell_state_decoherence(b, 1, 1, 150)


function compare_all_heatmaps(time=20,bell=get_bell(1,1),t1_range=10:10:200,t2_range=10:10:200)
    fig = Figure(size=(1800, 1200))
    
    # ADC parameters
    px_py_func(t,t1) = (1/4)*(1-exp(-t/t1))
    pz_func(t,t1,t2) = (1/2)*(1-exp(-t/t2)) - (1/4)*(1-exp(-t/t1))
    adc_lambda_func(t,t1,t2) = (px_py_func(t,t1), pz_func(t,t1,t2))
    adc_op_func = (t,t1,t2) -> begin
        px_py, pz = adc_lambda_func(t,t1,t2)
        ADCOp(1, px_py, pz)
    end
    manual_adc_op_func = (t,t1,t2) -> begin
        px_py, pz = adc_lambda_func(t,t1,t2)
        ManualADCOp(1, px_py, pz)
    end
  
    noise_funcs = [
        ((t,t1,t2) -> T1NoiseOp(1, λ₁_func(t, t1)), "T1NoiseOp", "T1 (Energy Relaxation)"),
        ((t,t1,t2) -> T2NoiseOp(1, λ₂_func(t, t1, t2)), "T2NoiseOp", "T2 (Dephasing)"),
        (adc_op_func, "ADCOp", "ADC (Amplitude Damping + Dephasing)"),
        (manual_adc_op_func, "ManualADCOp", "Manual ADC (Amplitude Damping + Dephasing)")
    ]
    # 2 by 2 = 4 heatmaps
    # fig already created above
    # title for the entire charts
    Label(fig[0, :], "Inconsistency between two $(1*time) μs and one $(2*time) μs op"; fontsize=16)

    for (idx, (gate_func, op_name, title_name)) in enumerate(noise_funcs)
        row, col = divrem(idx - 1, 2) .+ 1
        ax = Axis(fig[row, col]; 
                  xlabel="T2 (μs)", 
                  ylabel="T1 (μs)",
                  title="$(title_name) Inconsistency ($op_name)")
        
        map_2s_inconsistency(gate_func, time, t1_range, t2_range; bellstate=bell, ax=ax)
    end


    display(fig)
    return fig
end


# ============================================================================
# Run visualizations
# ============================================================================


SIMS = 200000
# SIMS =100

# fig = Figure(resolution=(800, 600))
fig = Figure()
ax = Axis(fig[1,1]; xlabel="T2 (μs)", ylabel="T1 (μs)", title="Inconsistency between 1-sec and 2-sec for T1NoiseOp at time=20 μs")
map_2s_inconsistency((t,t1,t2) -> T1NoiseOp(1, λ₁_func(t, t1)), 20, 10:10:200, 10:10:200; bellstate=get_bell(4,1), ax=ax)
Colorbar(fig[1,2], ax.scene.plots[1])
screen = display(fig)
# # # Uncomment to run:
# fig = 
    # plot_T1_results() # bad results
# #     # plot_T2_results() # good results
# #     # plot_ADC_results(;T1=30,T2=300) # ADC testing
# #     # plot_ADC_results(;T1=300,T2=20) # ADC testing
# #     # plot_ManualADC_results(;T1=100,T2=2000) # Manual ADC testing
# #     # plot_T1Depolarizing_results() # bad results
# #     # plot_all_noise_comparison()  # Compare all 4 noise types side by side
# #     # plot_bell_state_comparison(T1NoiseOp, t -> λ₁_func(t, 100); op_name="T1")
# #     # plot_bell_state_comparison(T2NoiseOp, t -> λ₂_func(t, 100, 150); op_name="T2")
# #     # plot_bell_state_comparison(T1DepolarizingOp, t -> λ₁_func(t, 100); op_name="T1Depolarizing")
# #     # plot_bell_state_comparison((idx, λ) -> ADCOp(idx, λ[1], λ[2]), 
# #     #                            t -> ((1/4)*(1-exp(-t/100)), (1/2)*(1-exp(-t/300)) - (1/4)*(1-exp(-t/100))); 
# #     #                            op_name="ADC", 
# #                             #    many_states=false)
# #     map_2s_inconsistency((t,t1,t2) -> T1NoiseOp(1, λ₁_func(t, t1)), 20, 10:10:200, 10:10:200; bellstate=get_bell(4,1))

# #     # compare_all_heatmaps(time=20, bell=get_bell(1,1), t1_range=10:10:200, t2_range=10:10:200)

# #     # compare_all_heatmaps(20, get_bell(1,1), 
# #     # 1:5:501,
# #     # 1:1:2,

# #     # #  1:5:501
# #     # 11:2
# #     #  )
                            
                               
# wait(display(fig))
# print("finished")
