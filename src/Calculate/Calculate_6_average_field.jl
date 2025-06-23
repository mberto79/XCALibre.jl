export calculate_field_average!
export Mean

struct Mean{T<:AbstractArray,I<:Integer}
    value::T
    start::I
    finish::I
end




function calculate_field_average!(mean_field::Mean,current_velocity,iteration)
    if mean_field.start <= iteration && iteration <= mean_field.finish
        n = iteration - mean_field.start + 1
        running_mean = (n-1)/n .* mean_field.value .+ Array(current_velocity) ./n
        mean_field.value .= running_mean
    end
    return nothing 
end






# function calculate_field_average!(mean_field::Mean,current_velocity,iteration)
#     if mean_field.start <= iteration && iteration <= mean_field.finish
#         Δ = Array(current_velocity) .- mean_field.velocity
#         current_average_velocity = (iteration-1)/iteration .* mean_field.velocity .+ Array(current_velocity) ./iteration
#         mean_field.velocity .= current_average_velocity
#         current_second_moment = mean_field.second_moment .+ (Δ .* (Array(current_velocity) .- current_average_velocity)) 
#         mean_field.second_moment .= current_second_moment
#     end
#     return nothing 
# end
