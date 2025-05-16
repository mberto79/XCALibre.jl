export calculate_field_average!
export Mean

struct Mean{T<:AbstractArray,I<:Integer}
    velocity::T
    second_moment::T 
    start::I
    finish::I
end

function calculate_field_average!(mean_field::Mean,current_velocity,iteration)
    if mean_field.start <= iteration && iteration <= mean_field.finish
        Δ = Array(current_velocity) .- mean_field.velocity
        current_average_velocity = (iteration-1)/iteration .* mean_field.velocity .+ Array(current_velocity) ./iteration
        mean_field.velocity .= current_average_velocity
        current_second_moment = mean_field.second_moment .+ (Δ .* (Array(current_velocity) .- current_average_velocity)) 
        mean_field.second_moment .= current_second_moment
    end
    return nothing 
end


