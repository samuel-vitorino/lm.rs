pub fn softmax(x: &mut [f32]){
    let mut sum: f32 = 0.0;

    for i in x.iter_mut() {
        *i = i.exp();
        sum += *i;
    }
    
    for i in x.iter_mut() {
        *i /= sum;
    }   
}