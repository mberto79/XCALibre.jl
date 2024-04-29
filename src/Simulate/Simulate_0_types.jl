export Configuration

@kwdef struct Configuration{SC,SL,RT}
    schemes::SC
    solvers::SL
    runtime::RT
end
Adapt.@adapt_structure Configuration
