/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.StringJoiner;
import java.util.function.Consumer;

/**
 * Column builder from Arrow data. 
 */
public final class ArrowColumnBuilder implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(ArrowColumnBuilder.class);

    private DType type;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private HostMemoryBuffer offsets;
    private long nullCount = 0l;
    //TODO nullable currently not used
    private boolean nullable;
    private long rows;
    private long estimatedRows;
    private String colName;
    private boolean built = false;
    private List<ArrowColumnBuilder> childBuilders = new ArrayList<>();

    private int currentIndex = 0;
    private int currentByteIndex = 0;


    public ArrowColumnBuilder(HostColumnVector.DataType type, long estimatedRows, String name) {
      this.type = type.getType();
      this.nullable = type.isNullable();
      this.colName = name;
      this.rows = 0;
      this.offsets = null;
      // TODO - rename estimatedRows to actual unless we do row count
      log.warn("in ArrowColVec colname is: " + name);
      this.estimatedRows = estimatedRows;
      for (int i = 0; i < type.getNumChildren(); i++) {
        childBuilders.add(new ArrowColumnBuilder(type.getChild(i), estimatedRows, name));
      }
    }

    public void setDataBuf(HostMemoryBuffer hdata) {
      data = hdata;
    }

    public void setValidityBuf(HostMemoryBuffer hdata) {
      valid = hdata;
    }

    public void setOffsetBuf(HostMemoryBuffer hdata) {
	// if (type.equals(DType.LIST) || type.equals(DType.STRING)) {
            offsets = hdata;
	// } else {
          //   throw new Exception("Error shouldn't be setting offset")
	// }
    }


    public ArrowColumnBuilder getChild(int index) {
      return childBuilders.get(index);
    }

    /**
     * Finish and create the immutable ColumnVector, copied to the device.
     */
    public final ColumnVector buildAndPutOnDevice() {
      log.warn("in buildAndPutOnDevice ArrowColVec");

      // try (ArrowHostColumnVector tmp = build()) {
	// TODO - fix to handle arrow
	// Create an ArrowTable for this column, then call from_arrow and then get ColumnVector
/*
  int type,
      String col_name,
      int col_length,
      int null_count,
      HostMemoryBuffer data,
      int data_size,
      HostMemoryBuffer validity,
      int validity_size,
      HostMemoryBuffer offsets
      int offsets_size)
*/
	// TODO - FIGURE OUT ROWS?
	log.warn("in buildAndPutOnDevice ArrowColVec 2");
        if (offsets == null) {
          log.warn("offsets is null 2");
        }
	log.warn("type before is: " + type + " name is: " + colName);
        ColumnVector vecRet = ColumnVector.fromArrow(type, colName, estimatedRows, nullCount, data, valid, offsets);
        log.warn("got vec to return type: " + vecRet.getType() + " lenght " + vecRet.getRowCount() + " is int: ");
	log.warn("back from fromArrow:" + vecRet.toString());
	
        return vecRet;
      // }
    }

    @Override
    public void close() {
      // memory buffers owned outside of this
      log.warn("in close 2 - do nothing");
      /*
      if (!built) {
        if (data != null) {
          data.close();
          data = null;
        }
        if (valid != null) {
          valid.close();
          valid = null;
        }
        if (offsets != null) {
          offsets.close();
          offsets = null;
        }
        for (ArrowColumnBuilder childBuilder : childBuilders) {
          childBuilder.close();
        }
        built = true;
      }
      */
    }

    @Override
    public String toString() {
      StringJoiner sj = new StringJoiner(",");
      for (ArrowColumnBuilder cb : childBuilders) {
        sj.add(cb.toString());
      }
      return "ArrowColumnBuilder{" +
          "type=" + type +
          ", children=" + sj.toString() +
          ", data=" + data +
          ", valid=" + valid +
          ", currentIndex=" + currentIndex +
          ", nullCount=" + nullCount +
          ", estimatedRows=" + estimatedRows +
          ", populatedRows=" + rows +
          ", built=" + built +
          '}';
    }
}
