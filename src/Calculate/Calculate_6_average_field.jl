export calculate_field_average!
export PostProcess

struct PostProcess{T<:AbstractScalarField,I<:Integer}
    field::T
    start::I
    finish::I
end


function calculate_field_average!(field_accumulator::PostProcess,current_field,iteration)
    if field_accumulator.start <= iteration && iteration <= field_accumulator.finish
        n = iteration - field_accumulator.start + 1
        running_mean = (n-1)/n .* field_accumulator.field.values .+ current_field ./n
        field_accumulator.field.values .= running_mean
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
