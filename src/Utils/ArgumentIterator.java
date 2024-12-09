/*
 * Author: Liam D. Tangney
 */

package Utils;

public class ArgumentIterator {
    String[] args = null;
    Integer index = null;

    public ArgumentIterator(String[] args) {
        this.args = args;
        this.index = 0;
    }

    public boolean hasNextFlag() {
        for (int i = this.index + 1; i < this.args.length; i++) {
            if (this.args[i].charAt(0) == '-') {
                return true;
            }
        }

        return false;
    }

    public boolean hasNextArgument() {
        for (int i = this.index + 1; i < this.args.length; i++) {
            if (this.args[i].charAt(0) != '-') {
                return true;
            }
        }

        return false;
    }

    public String nextFlag() {
        String flag = null;

        if (!this.hasNextFlag()) {
            return null;
        }

        if (this.index != 0) {
            this.index++;
        }

        while (this.index < this.args.length) {
            flag = this.args[this.index];
            if (flag.charAt(0) == '-') {
                return flag;
            }

            this.index++;
        }

        return null;
    }

    public String nextArgument() {
        String arg = null;

        if (!this.hasNextArgument()) {
            return null;
        }

        if (this.index != 0) {
            this.index++;
        }

        while (this.index < this.args.length) {
            arg = this.args[this.index];
            if (arg.charAt(0) != '-') {
                return arg;
            }

            this.index++;
        }

        return null;
    }
}
