export calculate_field_average!
export Mean

struct Mean{T<:AbstractArray}
    value::T 
end

function calculate_field_average!(mean_field::Mean,current_velocity,iteration)
    current_average = (iteration-1)/iteration .* mean_field.value .+ Array(current_velocity) ./iteration
    mean_field.value .= current_average
end
