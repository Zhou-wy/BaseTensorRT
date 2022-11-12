/*
 * @description: 
 * @version: 
 * @Author: zwy
 * @Date: 2022-10-13 20:24:12
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-16 13:48:39
 */

function clone(obj) {

    let o;
    if (typeof obj == "object") {
        if (obj === null) {
            o = null;
        } else {
            if (obj instanceof Array) {
                o = [];
                for (let i = 0, len = obj.length; i < len; i++) {
                    o.push(clone(obj[i]));
                }
            } else {
                o = {};
                for (let j in obj) {
                    o[j] = clone(obj[j]);
                }
            }
        }
    } else {
        o = obj;
    }
    return o;
}

const Tools = {
    clone
}


export default Tools