using BSON: @save

function save_model(model, path)
    model = model |> cpu
    @save path model
end