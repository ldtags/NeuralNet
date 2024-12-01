/*
 * Author: Liam D. Tangney
 */

public class ArgumentIterator {
    String[] args = null;
    Integer index = null;

    public ArgumentIterator(String[] args) {
        this.args = args;
        this.index = 0;
    }

    public boolean hasNext() {
        return index <= args.length;
    }

    public String getCurrentArgument() {
        if (!this.hasNext()) {
            return null;
        }

        return this.args[this.index];
    }

    public String nextFlag() {
        while (this.hasNext() && this.getCurrentArgument().charAt(0) != '-') {
            this.index++;
        }

        if (!this.hasNext()) {
            return null;
        }

        String arg = this.getCurrentArgument();
        this.index++;
        return arg;
    }

    public String nextArgument() {
        while (this.hasNext() && this.getCurrentArgument().charAt(0) == '-') {
            this.index++;
        }

        if (!this.hasNext() || this.getCurrentArgument().charAt(0) == '-') {
            return null;
        }

        String arg = this.getCurrentArgument();
        this.index++;
        return arg;
    }
}
